import os

import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset_LLM
from .transforms import AudioTransform, VideoTransform

IGNORE_INDEX = -100

def collate_LLM(batch, tokenizer, modality, is_trainval= True):

    pad_id = tokenizer.convert_tokens_to_ids('<pad>')
    
    batch_out = {}
    lengths = []
    # If we are in train/val mode, we use the entire text sequence. If we are in test mode, we only have the sos token. 
    
    if is_trainval:
        tokens = []
    if modality == "audio" or modality == "audiovisual" or modality == "audiovisual_avhubert":
        audios = [] if is_trainval else batch["audio"]
    if modality == "video" or modality == "audiovisual" or modality == "audiovisual_avhubert":
        videos = [] if is_trainval else batch["video"]
    
    if is_trainval:
        for i in range(len(batch)):
            tokens.append(batch[i]["tokens"])
            if modality == "audio" or modality == "audiovisual" or modality == "audiovisual_avhubert":
                audios.append(batch[i]["audio"])
                lengths.append(len(batch[i]["audio"]))
            if modality == "video" or modality == "audiovisual" or modality == "audiovisual_avhubert":
                videos.append(batch[i]["video"])
    else:
        if modality == "audio" or modality == "audiovisual" or modality == "audiovisual_avhubert":
            lengths.append(len(batch["audio"]))
    
    if tokenizer.name_or_path in ["TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf"]:
        tokens = tokenizer(tokens, padding= 'longest', return_tensors="pt").input_ids if is_trainval else torch.tensor([tokenizer.vocab["<s>"]]).unsqueeze(0)
    else:
        assert tokenizer.name_or_path == "meta-llama/Meta-Llama-3.1-8B"
        tokens = tokenizer(tokens, padding= 'longest', return_tensors="pt").input_ids if is_trainval else torch.tensor([tokenizer.vocab["<|begin_of_text|>"]]).unsqueeze(0)
    
    if is_trainval: # We need to set to -100 the padding tokens for the loss computation.
        labels = []
        for label in tokens:
            labels.append(torch.tensor([el if el != pad_id else IGNORE_INDEX for el in label], device = tokens.device))
            
    else: # In inference we don't have access to the text info.
        labels = None
        batch_out["gold_text"] = batch["tokens"]
        
    batch_out["tokens"] = tokens
    
    batch_out["labels"] = torch.stack(labels) if is_trainval else labels
    
    
    if modality == "audio" or modality == "audiovisual" or modality == "audiovisual_avhubert":
        audio_stack = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value = 0)
        batch_out["audio"] = audio_stack if is_trainval else audio_stack.unsqueeze(0)
        batch_out["lengths"] = torch.tensor(lengths)
        
    if modality == "video" or modality == "audiovisual" or modality == "audiovisual_avhubert":
        video_stack = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True, padding_value = 0)
        batch_out["video"] = video_stack if is_trainval else video_stack.unsqueeze(0)
    
    return batch_out


def _batch_by_token_count(idx_target_lengths, max_frames, batch_size=None):
    batches = []
    current_batch = []
    current_token_count = 0
    for idx, target_length in idx_target_lengths:
        if current_token_count + target_length > max_frames or (
            batch_size and len(current_batch) == batch_size
        ):
            batches.append(current_batch)
            current_batch = [idx]
            current_token_count = target_length
        else:
            current_batch.append(idx)
            current_token_count += target_length

    if current_batch:
        batches.append(current_batch)

    return batches


class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, lengths, max_frames, num_buckets, shuffle=False, batch_size=None
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [
            (idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)
        ]
        if shuffle:
            idx_length_buckets = random.sample(
                idx_length_buckets, len(idx_length_buckets)
            )
        else:
            idx_length_buckets = sorted(
                idx_length_buckets, key=lambda x: x[1], reverse=True
            )
        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class DataModule_LLM(LightningDataModule):
    def __init__(
        self,
        args,
        tokenizer,
        batch_size=None,
        train_num_buckets=50,
        train_shuffle=True,
        num_workers=5,
    ):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.train_num_buckets = train_num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.downsample_ratio = args.downsample_ratio_video
        self.is_avhubert_audio = True if args.modality == 'audiovisual_avhubert' or (args.audio_encoder_name == 'av-hubert' and args.modality =='audio') else False

    def train_dataloader(self):
        
        if self.args.modality == 'audio':
            self.args.max_frames = self.args.max_frames_audio
        elif self.args.modality == 'video':
            self.args.max_frames = self.args.max_frames_video
        else:
            self.args.max_frames = self.args.max_frames_audiovisual
        
        dataset = AVDataset_LLM(
            root_dir=self.args.root_dir,
            label_path=os.path.join(self.args.root_dir, "labels", self.args.train_file),
            subset="train",
            modality=self.args.modality,
            audio_transform=AudioTransform("train", is_avhubert_audio = self.is_avhubert_audio),
            video_transform=VideoTransform("train"),
            downsample_ratio=self.downsample_ratio,
            is_avhubert_audio = self.is_avhubert_audio,
            single_projector_avhubert = self.args.single_projector_avhubert,
        )
        
        dataset = CustomBucketDataset(
            dataset,
            dataset.input_lengths,
            self.args.max_frames,
            self.train_num_buckets,
            batch_size=self.batch_size,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
            collate_fn= lambda x: collate_LLM(x, self.tokenizer, self.args.modality, is_trainval= True),
        )
        return dataloader

    def val_dataloader(self):
        dataset = AVDataset_LLM(
            root_dir=self.args.root_dir,
            label_path=os.path.join(self.args.root_dir, "labels", self.args.val_file),
            subset="val",
            modality=self.args.modality,
            audio_transform=AudioTransform("val", is_avhubert_audio = self.is_avhubert_audio),
            video_transform=VideoTransform("val"),
            downsample_ratio=self.downsample_ratio,
            is_avhubert_audio = self.is_avhubert_audio,
            single_projector_avhubert = self.args.single_projector_avhubert,
        )
        dataset = CustomBucketDataset(
            dataset, dataset.input_lengths, 1000, 1, batch_size=self.batch_size
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            collate_fn= lambda x: collate_LLM(x, self.tokenizer, self.args.modality, is_trainval= True),
        )
        return dataloader

    def test_dataloader(self):
        dataset = AVDataset_LLM(
            root_dir=self.args.root_dir,
            label_path=os.path.join(self.args.root_dir, "labels", self.args.test_file),
            subset="test",
            modality=self.args.modality,
            audio_transform=AudioTransform(
                "test", snr_target=self.args.decode_snr_target,
                is_avhubert_audio = self.is_avhubert_audio,
                ),
            video_transform=VideoTransform("test"),
            downsample_ratio=self.downsample_ratio,
            is_avhubert_audio = self.is_avhubert_audio,
            single_projector_avhubert = self.args.single_projector_avhubert,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=None,
            collate_fn= lambda x: collate_LLM(x, self.tokenizer, self.args.modality, is_trainval= False),
        )
        return dataloader
