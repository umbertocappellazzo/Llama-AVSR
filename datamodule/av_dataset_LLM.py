import os
import torch
import torchaudio
import torchvision
import torch.nn.functional as F

from python_speech_features import logfbank
import numpy as np

def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data

def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)



# Note: since the number of tokens produced by audio encoders like WavLM and Whisper can vary a little, we can't truncate the
# audio samples here as we do for video such that we get an exact number of tokens compatible with the downsample ratios.

class AVDataset_LLM(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        downsample_ratio = None,
        is_avhubert_audio = False,
        single_projector_avhubert = None
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        
        self.is_avhubert_audio = is_avhubert_audio
        self.single_projector_avhubert = single_projector_avhubert
        
        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]
       
        if modality == "video" or modality == "audiovisual" or modality == "audiovisual_avhubert":
            self.downsample_video = downsample_ratio if downsample_ratio != 1 else None 
        
    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, _, text = path_count_label.split(",")
            paths_counts_labels.append((dataset_name, rel_path, input_length, text))
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, _, text = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            
            if self.downsample_video:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
            
            return {"video": video, "tokens": text}
        elif self.modality == "audio":
            audio = load_audio(path)
            
            audio = self.audio_transform(audio)
            
            if self.is_avhubert_audio:
                device = audio.device
                audio = logfbank(audio)
                audio = torch.tensor(stacker(audio, 4), dtype= torch.float32, device= device)
                with torch.no_grad():
                    audio = F.layer_norm(audio, audio.shape[1:])
                  
            return {"audio": audio, "tokens": text}
        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            
            if self.downsample_video:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
                
            return {"video": video, "audio": audio, "tokens": text}
        elif self.modality == "audiovisual_avhubert":
            assert self.is_avhubert_audio == True
            
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            
            video = self.video_transform(video)
            
            if not self.single_projector_avhubert:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
             
            device = audio.device
            audio = logfbank(audio)
            audio = torch.tensor(stacker(audio, 4), dtype= torch.float32, device= device)
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape[1:])
            return {"video": video, "audio": audio, "tokens": text}

    def __len__(self):
        return len(self.list)
