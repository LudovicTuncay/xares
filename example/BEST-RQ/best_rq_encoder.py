"""CHICA BEST-RQ adapter for xares.

This module exposes ``BestRQEncoder`` with the interface expected by xares:
  - ``output_dim`` attribute
  - ``forward(audio, audio_attention_mask) -> (embeddings, attention_mask)``

Example xares usage:
  python -m xares.run \
    example.BEST-RQ.best_rq_encoder.BestRQEncoder \
    esc50_task.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
import torchaudio

# Monkeypatch torchaudio.list_audio_backends if missing
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox", "ffmpeg"]

# Mock torchaudio.AudioMetaData if it is removed in new versions
if not hasattr(torchaudio, "AudioMetaData"):
    class AudioMetaData:
        def __init__(self, sample_rate, num_frames, num_channels, bits_per_sample, encoding):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding
    torchaudio.AudioMetaData = AudioMetaData

import torch.nn as nn
from torch import Tensor

DEFAULT_CHICASR_SRC_PATH = Path("/media/ltuncay/Shared-4TB/dev/chica-ai/chicasr/")
DEFAULT_CKPT_DPATH = Path(
    "/media/ltuncay/Shared-4TB/dev/chica-ai/chicasr/results/AudioSet_SSL_200k_bs800/2026-02-17_16.25.15/save/CKPT+2026-02-18_07.24.50_epoch_8_step_200000_loss_ce_AudioSet_eval_soxrhq_2.466886411313218"
)
FALLBACK_CKPT_DPATH = Path(
    "/media/ltuncay/Shared-4TB/dev/chica-ai/chicasr/results/AudioSet_SSL_200k_bs800/2026-02-17_16.25.15/save/CKPT+2026-02-18_00.08.20_epoch_4_step_103836_loss_ce_AudioSet_eval_soxrhq_2.3985468574667927"
)

ValidateChecksum = Literal["ignore", "warn", "strict"]


def _to_bool_mask(mask: Tensor) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point():
        return mask > 0.5
    return mask > 0


def _validate_ckpt_layout(ckpt_dpath: Path) -> None:
    if not ckpt_dpath.is_dir():
        raise FileNotFoundError(
            f"Invalid ckpt_dpath='{ckpt_dpath}': directory does not exist."
        )

    required_in_ckpt = (
        "CKPT.yaml",
        "model.ckpt",
        "normalize.ckpt",
        "scheduler.ckpt",
        "counter.ckpt",
        "quantizer.ckpt",
        "linear.ckpt", 
    )
    missing = [
        name for name in required_in_ckpt if not ckpt_dpath.joinpath(name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Checkpoint directory is missing required files: "
            f"{missing}. Expected inside '{ckpt_dpath}'."
        )

    # CHICA loader expects ../../hyperparams.yaml from ckpt dir.
    hparams_fpath = ckpt_dpath.parent.parent.joinpath("hyperparams.yaml")
    if not hparams_fpath.is_file():
        raise FileNotFoundError(
            "Cannot find CHICA hyperparameters file at "
            f"'{hparams_fpath}'. Ensure ckpt_dpath points to a CHICA SSL CKPT+... directory."
        )


class BestRQEncoder(nn.Module):
    def __init__(
        self,
        ckpt_dpath: str | None = None,
        device_name: str = "cuda_if_available",
        validate_checksum: ValidateChecksum = "warn",
        chicasr_src_path: str = str(DEFAULT_CHICASR_SRC_PATH),
    ) -> None:
        super().__init__()

        self.ckpt_dpath = Path(ckpt_dpath) if ckpt_dpath else DEFAULT_CKPT_DPATH
        self.chicasr_src_path = Path(chicasr_src_path)
        _validate_ckpt_layout(self.ckpt_dpath)

        # Make local CHICA source importable in xares runtime.
        if str(self.chicasr_src_path) not in sys.path:
            sys.path.append(str(self.chicasr_src_path))

        try:
            from chicasr.ssl.core import pad_feats
            from chicasr.ssl.infer import load_ssl_brain_form_ckpt_dpath
        except Exception as exc:
            raise RuntimeError(
                "Failed to import CHICA SSL runtime modules. "
                "Install CHICA dependencies (e.g. torchwrench, pythonwrench, speechbrain, hyperpyyaml, lejepa) "
                "in the Python environment running xares."
            ) from exc

        self._pad_feats = pad_feats
        self.brain = load_ssl_brain_form_ckpt_dpath(
            ckpt_dpath=self.ckpt_dpath,
            device_name=device_name,
            validate_checksum=validate_checksum,
        )
        self.hparams = self.brain.hparams
        self.sb_modules = self.brain.modules

        self.brain.modules.eval()
        for param in self.brain.modules.parameters():
            param.requires_grad_(False)

        # Prefer static, meta-safe inference from the CHICA pre-classification head input.
        linear_w = getattr(getattr(self.sb_modules, "linear", None), "w", None)
        in_features = getattr(linear_w, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            self.output_dim = int(in_features)
        else:
            # Fallback: infer with a tiny dry forward pass.
            self.output_dim = int(self._infer_output_dim())

        self.sampling_rate = int(self.hparams.sample_rate)
        
        # Calculate hop_size_in_ms using a dry run
        dummy_wavs = torch.zeros(1, self.sampling_rate, device=self.device)
        dummy_props = torch.ones(1, device=self.device)
        enc = self._compute_encoder(dummy_wavs, dummy_props)
        # T' is the number of output frames for 1 second of audio
        out_frames = enc.shape[1]
        self.hop_size_in_ms = 1000.0 / out_frames

    @property
    def device(self) -> torch.device:
        try:
            return next(self.sb_modules.parameters()).device
        except StopIteration:
            return torch.device(self.brain.device)

    @torch.no_grad()
    def _infer_output_dim(self) -> int:
        sr = int(self.hparams.sample_rate)
        # 1 second dry sample is enough to infer channel dim.
        dummy_wavs = torch.zeros(1, sr, device=self.device)
        dummy_props = torch.ones(1, device=self.device)
        enc = self._compute_encoder(dummy_wavs, dummy_props)
        if enc.ndim != 3:
            raise RuntimeError(
                f"Unexpected encoder output shape while inferring dim: {tuple(enc.shape)}"
            )
        return int(enc.shape[-1])

    @torch.no_grad()
    def _compute_encoder(self, wavs: Tensor, wav_props: Tensor) -> Tensor:
        current_epoch = int(self.hparams.epoch_counter.current)
        feats = self.hparams.compute_features(wavs)
        feats = self.sb_modules.normalize(feats, wav_props, epoch=current_epoch)
        feats = self._pad_feats(feats, int(self.hparams.pad_to_divisible_by))
        src = self.sb_modules.CNN(feats)
        # IMPORTANT: we intentionally stop at the encoder wrapper output.
        # We do NOT apply `self.sb_modules.linear`, which is the pre-classification head.
        enc = self.sb_modules.wrapper(src, wav_props)
        return enc

    @torch.no_grad()
    def forward(
        self, audio: Tensor, audio_attention_mask: Tensor | None = None
    ) -> Tensor:
        # xares expects [B, T]; accept [T] too.
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise ValueError(
                f"Expected audio with shape [B, T] or [T], got {tuple(audio.shape)}"
            )

        audio = audio.to(self.device)
        batch_size, num_samples = audio.shape

        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                batch_size, num_samples, dtype=torch.long, device=audio.device
            )
        else:
            if audio_attention_mask.ndim == 1:
                audio_attention_mask = audio_attention_mask.unsqueeze(0)
            if tuple(audio_attention_mask.shape) != (batch_size, num_samples):
                raise ValueError(
                    "audio_attention_mask shape mismatch: "
                    f"expected {(batch_size, num_samples)}, got {tuple(audio_attention_mask.shape)}"
                )
            audio_attention_mask = audio_attention_mask.to(audio.device)

        split_size = int(20 * self.sampling_rate)
        audio_splits = torch.split(audio, split_size, dim=-1)
        mask_splits = torch.split(audio_attention_mask, split_size, dim=-1)

        enc_splits = []
        for a_split, m_split in zip(audio_splits, mask_splits):
            split_num_samples = a_split.shape[1]
            valid_sample_mask = _to_bool_mask(m_split)
            valid_lengths = valid_sample_mask.sum(dim=1).clamp(min=1)
            wav_props = valid_lengths.to(dtype=torch.float32) / float(split_num_samples)

            enc_split = self._compute_encoder(a_split, wav_props)
            if enc_split.ndim != 3:
                raise RuntimeError(
                    f"Expected encoder output [B, T_enc, D], got {tuple(enc_split.shape)}"
                )
            enc_splits.append(enc_split)

        enc = torch.cat(enc_splits, dim=1)

        return enc


if __name__ == "__main__":
    ckpt = DEFAULT_CKPT_DPATH
    model = BestRQEncoder(ckpt_dpath=str(ckpt))
    print(f"Initialized BestRQEncoder with output_dim={model.output_dim}")
    sample = torch.randn(2, 48000)
    sample_mask = torch.ones_like(sample, dtype=torch.long)
    sample_mask[1, 24000:] = 0
    out, out_mask = model(sample, sample_mask)
    print("out shape:", tuple(out.shape))
    print("mask shape:", tuple(out_mask.shape))
