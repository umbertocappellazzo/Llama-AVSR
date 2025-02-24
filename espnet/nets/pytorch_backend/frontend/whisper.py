import torch
import whisper
from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES

N_MELS = 80
HOP_LENGTH = 320

class WhisperFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS
        self.mel_filters = whisper.audio.mel_filters

    def forward(self, audio, ilens=None):
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(audio, self.n_fft, self.hop_length, window=window, return_complex=True)

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2
        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec, olens
