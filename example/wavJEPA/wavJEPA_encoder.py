import torch
from transformers import AutoFeatureExtractor, AutoModel


class WavJEPAEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.output_dim = 768
        self.hop_size_in_ms = 10
        self.model = AutoModel.from_pretrained(
            "labhamlet/wavjepa-base", trust_remote_code=True
        )
        self.extractor = AutoFeatureExtractor.from_pretrained(
            "labhamlet/wavjepa-base", trust_remote_code=True
        )

    def forward(self, audio: torch.Tensor):
        assert isinstance(audio, torch.Tensor)
        extracted = self.extractor(audio, return_tensors="pt")

        self.model.eval()
        with torch.inference_mode():
            output = self.model(extracted["input_values"])[0]
        return output


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    encoder = WavJEPAEncoder()
    assert check_audio_encoder(encoder)
