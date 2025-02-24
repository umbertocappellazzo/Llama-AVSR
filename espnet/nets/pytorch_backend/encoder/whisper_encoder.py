import copy
import torch
import whisper


class OpenAIWhisperEncoder(torch.nn.Module):
    def __init__(self, whisper_model="small"):
        super().__init__()
        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(whisper_model)
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoder_size = _model.dims.n_audio_state
        del _model

    def whisper_encode(self, input, ilens=None):
        x = torch.nn.functional.gelu(self.encoders.conv1(input))
        x = torch.nn.functional.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        for layer, block in enumerate(self.encoders.blocks):
            x = block(x)
        x = self.encoders.ln_post(x)

        if ilens is not None:
            olens = (1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(self, xs_pad, ilens=None):
        xs_pad, olens = self.whisper_encode(xs_pad, ilens)
        return xs_pad, olens
