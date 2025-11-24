import logging
import os
from argparse import ArgumentParser
import torch
import time
from utils.avg_checkpoints_original import ensemble_original
from datamodule.data_module import DataModule_LLM
from models.lightning import ModelModule_LLM

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy#, FSDPStrategy, DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_trainer(args):
    seed_everything(args.seed, workers=True)  # Default seed: 42. Alternative: 7.
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=False,
        filename="{epoch}",
        save_top_k=args.num_check_save, 
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    find_unused_parameters_flag = False if args.modality == 'audio' else True

    return Trainer(
        precision='bf16-true',
        sync_batchnorm=True,
        num_sanity_val_steps=2,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy= DDPStrategy(find_unused_parameters= find_unused_parameters_flag),  
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        logger=WandbLogger(name=args.exp_name, project=args.project_wandb),
        gradient_clip_val=10.0,
        val_check_interval=args.val_check_interval,
    )


def get_test_trainer(args):
    return Trainer(precision='bf16-true',
        num_nodes=1,
        devices=1,
        accelerator="gpu",
        logger=WandbLogger(name=args.exp_name, project=args.project_wandb),
    )

def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default=None,
        type=str
    )
    parser.add_argument(
        "--root-dir",
        default=None ,
        type=str,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--project-wandb",
        default=None ,
        type=str,
        help="Name of the wandb project.",
    )
    parser.add_argument(
        "--exp-name",
        default= "", 
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--modality",
        default=None,
        type=str,
        help="Type of input modality",
        choices=["audio", "video", "audiovisual", "audiovisual_avhubert"],
    )
    parser.add_argument(
        "--llm-model",
        default= None,
        type=str,
        help="LLM model name",
        choices= ["TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-13b-hf", 
                  "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3.1-8B"
                 ]
    )
    parser.add_argument(
        "--intermediate-size",
        default= 2048,
        type=int,
        help="Intermediate size of the projector.",
    )
    parser.add_argument(
        "--single-projector-avhubert",
        default= False,
        type=bool,
        help="""This parameter is used only when modality == audiovisual_avhubert. If set to True, a single audio-visual projector
                is trained on top of the audio-visual features output by AV-HuBERT. If set to False, audio and video features
                are computed twice with AV-HuBERT with the other modality set to None""",
    )
    parser.add_argument(
        "--prompt-audio",
        default= "Transcribe speech to text.",
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-video",
        default= "Transcribe video to text.",
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-audiovisual",
        default= "Transcribe speech and video to text.",
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-video-path",
        default= None,
        type=str
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-audio-path",
        default= None,  
        type=str,
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-audiovisual-path",
        default= None,
        type=str,
    )
    parser.add_argument(
        "--use-lora-avhubert",
        default = False,
        type = bool,
        help= "Whether to apply LoRA to the transformer module of AV-HuBERT."
        )
    parser.add_argument(
        "--audio-encoder-name",
        default = None, # "openai/whisper-medium.en/small.en/base.en/tiny.en/large",   "microsoft/wavlm-large"
        type = str,
        choices= ["openai/whisper-medium.en", "microsoft/wavlm-large", "av-hubert"]
        )
    parser.add_argument(
        "--unfrozen_modules",
        nargs="*",
        default= [None], #  "peft_llm","lora_avhubert"
        help="Which modules to train.",
        choices = [None, "peft_llm", "lora_avhubert"]
    )
    parser.add_argument(
        "--add_PETF_LLM",
        default= None,
        type= str,
        help="Whether to add a PEFT module to the LLM.",
        choices= [None, "lora"]
    )
    parser.add_argument(
        "--reduction_lora",
        default= None,
        type=int,
        help="Rank for LoRA."
    )
    parser.add_argument(
        "--alpha",
        default= None,
        type=int,
        help="Alpha for LoRA."
    )
    parser.add_argument(
        "--downsample-ratio-audio",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-audiovisual",
        default=2,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--train-file",
        default="lrs3_train_transcript_lengths_seg16s_LLM_lowercase_greater25.csv",
        type=str,
        help="Filename of training label list",
    )
    parser.add_argument(
        "--val-file",
        default="lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv",
        type=str,
        help="Filename of validation label list.",
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv",
        type=str,
        help="Filename of testing label list.",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of machines used. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus in each machine. (Default: 8)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default= None,
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs for warmup. (Default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num-average-epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num-check-save",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--val-check-interval",
        default=1.,
    )
    parser.add_argument(
        "--max-frames-audio",
        type=int,
        default=1000,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--max-frames-video",
        type=int,
        default=1500,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--max-frames-audiovisual",
        type=int,
        default=1000,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,   # 1e-3 for ASR and AVSR, 5e-4 for VSR.
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path of the checkpoint from which training is resumed.",
    )
    
# Inference parameters. 
    parser.add_argument(
        "--max-dec-tokens",
        default= 32,
        type=int,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-beams",
        default=15,
        type=int,
        help="Beams used for beam search",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=float,
        default=-1,
        help="Slurm job id",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default= 999999,  
        help="Level of signal-to-noise ratio (SNR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    parser.add_argument(
        "--auto-test",
        default= True,
        help="Flag to use debug level for logging",
    )
    parser.add_argument(
        "--add-sink-loss",
        type=bool,
        default=False,
        help="Add decorrelation loss to avoid intermediate attention sinks",
    )
    parser.add_argument(
        "--sink-loss-factor",
        type=float,
        default=10000,
        help="Weight of sink loss in reference to cross entropy loss. Note: add-sink-loss must be True",
    )
    parser.add_argument(
        "--layernorm-projector",
        default=False,
        type=bool,
        help="Removes LayerNorm from the audio and video projectors",
    )
    
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    print(args)
    
    if args.slurm_job_id != -1:
        args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    
    modelmodule = ModelModule_LLM(args)
    datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.print(torch.cuda.memory_summary())
    
    if args.auto_test:
        
        args.pretrained_model_path = ensemble_original(args, args.num_average_epochs)
        time.sleep(600)
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = get_test_trainer(args)
            ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            modelmodule.model.load_state_dict(ckpt)
            
            trainer.test(model=modelmodule, datamodule=datamodule)
if __name__ == "__main__":
    cli_main()
