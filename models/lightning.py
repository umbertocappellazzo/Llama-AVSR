import sys
sys.path.append("..")
import torch
import torchaudio
from utils.cosine import WarmupCosineScheduler
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from .Llama_LoRA import LoRA_config
from .modeling_AVSRLLM import AVSR_LLMs
from tokenizers.processors import TemplateProcessing

DEFAULT_PAD_TOKEN = "<pad>"
AUDIO_SOS = "<audio>"
AUDIO_EOS = "</audio>"
VIDEO_SOS = "<video>"
VIDEO_EOS = "</video>"

llm_size = {"TinyLlama/TinyLlama_v1.1": 2048,
            "meta-llama/Llama-2-13b-hf": 5120,
            "meta-llama/Llama-2-7b-hf": 4096,
            "meta-llama/Meta-Llama-3.1-8B": 4096,
            }


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)

def compute_word_level_edits(seq1, seq2, include_wer_breakdown):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    if not include_wer_breakdown:
        return {"total_cost": torchaudio.functional.edit_distance(seq1, seq2)}
    len_sent2 = len(seq2)
    dold = [(j, j, 0, 0) for j in range(len_sent2+1)]
    dnew = [(0,0,0,0) for _ in range(len_sent2 + 1)] # distance, insertions, deletions, substitutions

    num_edits = {'substitutions': 0, 'insertions': 0, 'deletions': 0}

    for i in range(1, len(seq1) + 1):
        dnew[0] = (i, i, 0, 0)
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                sub_cost, ins_s, del_s, sub_s = dold[j-1]
                sub_opt = (sub_cost+1, ins_s, del_s, sub_s+1)

                ins_cost, ins_i, del_i, sub_i = dnew[j-1]
                ins_opt = (ins_cost+1, ins_i+1, del_i, sub_i)
                
                del_cost, ins_d, del_d, sub_d = dold[j]
                del_opt = (del_cost+1, ins_d, del_d+1, sub_d)
                
                dnew[j] = min(sub_opt, ins_opt, del_opt, key=lambda x: x[0])

	
        dold, dnew = dnew, dold

        total_cost, ins, dels, subs = dold[-1]
    
    return {"total_cost": total_cost, "insertions": ins, "deletions": dels, "substitutions": subs}

class ModelModule_LLM(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.include_wer_breakdown = args.include_wer_breakdown
        self.edits = {"total_cost": 0, 'substitutions': 0, 'insertions': 0, 'deletions': 0} if self.include_wer_breakdown else {"total_cost": 0}
        
        if args.use_lora_avhubert:
            assert "lora_avhubert" in args.unfrozen_modules, ("LoRA modules for the AV-HuBERT encoder must be unfrozen!!")
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, add_bos_token=True, add_eos_token= True)
        
        # Apparently, some LLMs don't rely on FastTokenizer and it seems like they don't append the EOS token even though you set
        # it explicitly. In my case, this happens for LLama3. More details at: https://github.com/huggingface/transformers/issues/22794.
        
        if args.llm_model == "meta-llama/Meta-Llama-3.1-8B":
            bos = self.tokenizer.bos_token
            eos = self.tokenizer.eos_token
            
            self.tokenizer._tokenizer.post_processor =TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[
                    (f"{bos}", self.tokenizer.bos_token_id), 
                    (f"{eos}", self.tokenizer.eos_token_id)
                    ],
                )
        
        # By default, LLaMA doesn't come with a padding token (pad_token= None), so we need to introduce it.
        num_added_toks = self.tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN, "additional_special_tokens": [AUDIO_SOS, AUDIO_EOS, VIDEO_SOS, VIDEO_EOS]})
        pad_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
       
            
        print("We have added ", num_added_toks, " tokens to the tokenizer!")
        self.tokenizer.padding_side = "right"   # The padding is added to the right.
        
        # The resize of the embed_tokens matrix and the add of the pad_token to the model is performed when the model is called.
        
        if args.modality == 'audio':
            prompt = args.prompt_audio
        elif args.modality == 'video':
            prompt = args.prompt_video
        else:
            assert args.modality in ['audiovisual', 'audiovisual_avhubert']
            prompt = args.prompt_audiovisual
        
        print(f"The prompt used for the {args.modality} modality is: {prompt}")
        
        if args.add_PETF_LLM:
            
            IS_LLAMA3 = True if args.llm_model == "meta-llama/Meta-Llama-3.1-8B" else False
            IS_TINYLLAMA = True if args.llm_model == "TinyLlama/TinyLlama_v1.1" else False
            lora_config_llm = LoRA_config(args.reduction_lora, args.alpha, IS_LLAMA3, IS_TINYLLAMA)
            
            self.model = AVSR_LLMs(modality = args.modality,  
                                   pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path, 
                                   pretrain_avhubert_enc_audio = args.pretrain_avhubert_enc_audio_path, 
                                   pretrain_avhubert_enc_audiovisual = args.pretrain_avhubert_enc_audiovisual_path,
                                   use_lora_avhubert= args.use_lora_avhubert,
                                   llm_model = args.llm_model, 
                                   hidden_size = llm_size[args.llm_model], 
                                   intermediate_size= args.intermediate_size, 
                                   tokenizer = self.tokenizer, 
                                   prompt = prompt, 
                                   pad_id = pad_id, 
                                   downsample_ratio_audio = args.downsample_ratio_audio, 
                                   downsample_ratio_video = args.downsample_ratio_video, 
                                   downsample_ratio_audiovisual = args.downsample_ratio_audiovisual,
                                   single_projector_avhubert = args.single_projector_avhubert,
                                   audio_encoder_name = args.audio_encoder_name,
                                   unfrozen_modules= args.unfrozen_modules, 
                                   max_dec_tokens = args.max_dec_tokens, 
                                   num_beams = args.num_beams, 
                                   PETF_LLM_name = args.add_PETF_LLM, 
                                   peft_config_llm= lora_config_llm, 
                                   )
            
            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)
        
        else:
            self.model = AVSR_LLMs(modality = args.modality, 
                                   pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path,
                                   pretrain_avhubert_enc_audio = args.pretrain_avhubert_enc_audio_path, 
                                   pretrain_avhubert_enc_audiovisual = args.pretrain_avhubert_enc_audiovisual_path,
                                   use_lora_avhubert= args.use_lora_avhubert,
                                   llm_model = args.llm_model,
                                   hidden_size = llm_size[args.llm_model],
                                   intermediate_size= args.intermediate_size,
                                   tokenizer = self.tokenizer,
                                   prompt = prompt,
                                   pad_id = pad_id,
                                   downsample_ratio_audio = args.downsample_ratio_audio,
                                   downsample_ratio_video = args.downsample_ratio_video,
                                   downsample_ratio_audiovisual = args.downsample_ratio_audiovisual,
                                   single_projector_avhubert = args.single_projector_avhubert,
                                   audio_encoder_name = args.audio_encoder_name,
                                   unfrozen_modules= args.unfrozen_modules,
                                   max_dec_tokens = args.max_dec_tokens,
                                   num_beams = args.num_beams, 
                                   )
            
            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)

            
            
        
        # initialize the full model from the checkpoint for inference.
        if args.pretrained_model_path:
            ckpt = torch.load(args.pretrained_model_path)
            self.model.load_state_dict(ckpt)
            
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params= self.model.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        train_loss = self.model(batch, is_trainval = True)[0]
        
        batch_size = batch["tokens"].shape[0]

        self.log("loss", train_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        
        batch_sizes = self.all_gather(batch_size)
        
        train_loss *= batch_sizes.size(0) / batch_sizes.sum()
        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
        return train_loss
            
        
    def validation_step(self, batch, batch_idx):
        val_loss = self.model(batch, is_trainval = True)[0]
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss_val", val_loss, batch_size=batch_size, sync_dist=True)
        
        return val_loss
            
    def test_step(self, batch, batch_idx):
        
        generated_ids = self.model(batch, is_trainval = False)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print("Input text: ", batch["gold_text"])
        print("Generated text: ", generated_text)

        edit_updates = compute_word_level_edits(batch["gold_text"], generated_text, self.include_wer_breakdown)
        self.edits = {k: edit_updates[k] + self.edits[k] for k in set(self.edits)}
        self.total_length += len(batch["gold_text"].split())

        return
    
    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        
    def on_test_epoch_end(self):
        self.log("wer", self.edits['total_cost'] / self.total_length)
        if self.include_wer_breakdown:
                self.log("word substitution rate", self.edits['substitutions'] / self.total_length)
                self.log("word insertion rate", self.edits['insertions'] / self.total_length)
                self.log("word deletion rate", self.edits['deletions'] / self.total_length)
