"""
X ARES compatible wrapper around the JEPA vision transformer encoder.

Usage
-----
>>> from xares.audio_encoder_checker import check_audio_encoder
>>> encoder = JEPAEncoder(checkpoint="jepa.ckpt", device="cuda")
>>> check_audio_encoder(encoder)      # should print True
"""

from pathlib import Path
from typing import List, Optional
import math

import torch
import torch.nn.functional as F

# Local import (relative to example/jepa/)
from .components.vision_transformer import VisionTransformer
# from .audio_prepoc import normalize_audio, pad, MelSpecTransform

from .utils.audio_preprocessing import normalize_audio, MelSpecTransform


class JEPAEncoder(torch.nn.Module):
    """Convert raw audio into a sequence of JEPA ViT embeddings.

    The class obeys the interface expected by `xares.audio_encoder_checker`:

    * ``sampling_rate``: integer Hz, the required input sampling rate.
    * ``output_dim``:    dimensionality of each output vector (D).
    * ``hop_size_in_ms``: real, time shift (ms) between two consecutive output
      vectors.

    The encoder internally
    1. splits audio longer than 10 s into non-overlapping 10 s chunks;
    2. pads the last chunk with zeros up to 10 s (if needed);
    3. converts each chunk to a (1 x 256 x 128) log mel spectrogram;
    4. feeds the spectrogram through a ViT with patch size 16x16 (TxF);
    5. mean pools the 8 frequency patches inside every time patch, producing
       *one* vector per time patch;
    6. removes the extra time patches coming from padding if the chunk was
       shorter than 10 s.

    The final output has shape ``[B, T, D]`` where ``T`` is proportional to the
    true audio duration (16 vectors per 10 s).
    """

    def __init__(
        self,
        checkpoint: Optional[str] = r"example/jepa/checkpoints/JEPA.ckpt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        # ---- constants -------------------------------------------------- #
        self.device = torch.device(device)
        self.sampling_rate: int = 32_000           # required by X‑ARES
        self.clip_length_sec: int = 10             # JEPA was trained on 10 s clips
        self.clip_samples: int = self.clip_length_sec * self.sampling_rate
        self.target_time_bins: int = 256           # Mel‑spec frames per 10 s
        self.n_mels: int = 128

        # ViT patching parameters (time, freq)
        self.patch_size_t: int = 16                # 256 // 16 = 16 patches in T
        self.patch_size_f: int = 16                # 128 // 16 = 8  patches in F
        self.num_time_patches: int = self.target_time_bins // self.patch_size_t  # 16
        self.num_freq_patches: int = self.n_mels // self.patch_size_f            # 8

        # Each time‑patch corresponds to 10 s / 16 = 0.625 s = 625 ms
        self.hop_size_in_ms: float = (self.clip_length_sec * 1_000) / self.num_time_patches

        # ---- preprocessing --------------------------------------------- #
        self.mel_transform = MelSpecTransform(
            sr=self.sampling_rate,
            n_mels=self.n_mels,
            clip_length=self.clip_length_sec,
            target_time_bins=self.target_time_bins,
            log=True,
        ).to(self.device)

        # ---- JEPA backbone --------------------------------------------- #
        self.encoder = VisionTransformer(
            patch_size=(self.patch_size_t, self.patch_size_f),
            in_chans=1,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
        ).to(self.device)

        self.output_dim: int = 768

        # -------- load checkpoint (weights_only=True) --------------------- #
        if checkpoint is not None:
            ckpt_path = Path(checkpoint).expanduser()
            if not ckpt_path.is_file():
                raise FileNotFoundError(ckpt_path)

            state = torch.load(ckpt_path, map_location=self.device)
            if "state_dict" in state:  # lightning checkpoints
                state = state["state_dict"]

            # JEPA checkpoints prefix the online/target encoders
            prefix = "target_encoder."
            if not any(k.startswith(prefix) for k in state.keys()):
                prefix = "encoder."

            state = {
                k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
            }
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys when loading checkpoint: {unexpected}")

        # put entire module in eval
        self.eval()

        # ensure transforms and encoder use inference mode
        self.mel_transform.eval()
        self.encoder.eval()

    # ---------------------------------------------------------------------
    # helper functions
    # ---------------------------------------------------------------------
    def _waveform_to_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """Return (1 x 256 x 128) log-mel spectrogram on *self.device*."""
        if wav.ndim != 1:
            raise ValueError("Expected mono waveform [T]")

        wav = normalize_audio(wav)                 # (T)
        spec = self.mel_transform(wav.unsqueeze(0))  # (1, T_bins, n_mels)
        return spec.to(self.device)                # (1, 256, 128)

    def _spectrogram_to_patch_embeddings(self, spec: torch.Tensor) -> torch.Tensor:
        """Run ViT and mean-pool over frequency patches → (16, D)."""
        # ViT expects (B, C, H, W)
        x = spec.unsqueeze(0)                      # (1, 1, 256, 128)

        with torch.inference_mode():
            feats = self.encoder(x)                # (1, N[+1], D)

        # Remove CLS if present
        if feats.shape[1] == (self.num_time_patches * self.num_freq_patches + 1):
            feats = feats[:, 1:, :]

        # (1, 16*8, D) → (1, 16, 8, D)
        feats = feats.view(1, self.num_time_patches, self.num_freq_patches, -1)
        feats = feats.mean(dim=2)                  # mean‑pool over frequency → (1, 16, D)
        return feats.squeeze(0)                    # (16, D)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """Encode *audio* waveforms into JEPA patch embeddings.

        Parameters
        ----------
        audio: ``torch.Tensor``
            Shape ``[B, T]`` (or ``[T]``). Must be self.sampling_rate kHz.

        Returns
        -------
        torch.Tensor
            Shape ``[B, T_out, D]`` where ``T_out = ceil(duration/10s) * 16``.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # → [1, T]

        B, total_len = audio.shape
        audio = audio.to(self.device)

        all_outputs: List[torch.Tensor] = []
        for b in range(B):
            wav = audio[b]
            n_samples = wav.shape[0]
            n_chunks = math.ceil(n_samples / self.clip_samples)
            sample_embeds: List[torch.Tensor] = []

            for c in range(n_chunks):
                start = c * self.clip_samples
                end = min(start + self.clip_samples, n_samples)
                chunk = wav[start:end]
                valid_samples = chunk.shape[0]

                # zero‑pad up to 10 s
                if valid_samples < self.clip_samples:
                    chunk = F.pad(chunk, (0, self.clip_samples - valid_samples))

                # mel‑spec and ViT
                spec = self._waveform_to_spectrogram(chunk)
                patch_embeds = self._spectrogram_to_patch_embeddings(spec)  # (16, D)

                # keep only the patches that overlap the real audio
                ratio = valid_samples / self.clip_samples
                valid_t = int(math.ceil(ratio * self.num_time_patches))
                patch_embeds = patch_embeds[:valid_t]
                sample_embeds.append(patch_embeds)

            all_outputs.append(torch.cat(sample_embeds, dim=0))  # (T_i, D)

        # Pad to max‑length so we can return a single tensor
        max_T = max(t.shape[0] for t in all_outputs)
        output = audio.new_zeros((B, max_T, self.output_dim))
        for b, emb in enumerate(all_outputs):
            output[b, : emb.shape[0]] = emb

        return output




# # ------------------------------------------------------------------------- #
# class JEPAEncoder(torch.nn.Module):
#     sampling_rate: int = 32_000                                             # required by xares
#     hop_size_in_ms: int = 10                                                # required by xares
#     output_dim: int = 768                                                   # set after model is built
#     clip_length: int = 10                                                   # seconds
#     custom_max_audio_length: int = int(sampling_rate * clip_length)         # 320000 samples

#     def __init__(self, checkpoint: Optional[str] = None, device: str | torch.device = "cpu"):
#         super().__init__()
#         self.device = torch.device(device)

#         self.mel = MelSpecTransform(
#             sr=self.sampling_rate,
#             n_mels=128,
#             clip_length=10,
#             target_time_bins=256,
#             log=True,
#         ).to(self.device)

#         self.encoder = VisionTransformer(
#             patch_size=(16, 16),
#             in_chans=1,
#             embed_dim=768,
#             depth=12,
#             num_heads=12,
#             mlp_ratio=4.0,
#         ).to(self.device)

#         self.output_dim = self.encoder.embed_dim

#         # -------- load checkpoint (weights_only=True) --------------------- #
#         if checkpoint is not None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             ckpt_path = Path(checkpoint).expanduser()
#             state = torch.load(ckpt_path, map_location=device, weights_only=True)

#             if "state_dict" in state:
#                 state = state["state_dict"]

#             prefix = "target_encoder."
#             if not any(k.startswith(prefix) for k in state):
#                 prefix = "encoder."                         # <- graceful fallback


#             state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

#             missing, unexpected = self.encoder.load_state_dict(state, strict=False)
#             if unexpected:
#                 raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")

#         self.encoder.eval()

#     # # ---------------------------------------------------------------------- #
#     # @torch.inference_mode()
#     # def forward(self, audio: torch.Tensor) -> torch.Tensor:                  # noqa: D401
#     #     """
#     #     Parameters
#     #     ----------
#     #     audio : `torch.Tensor`
#     #         `[T]` or `[B,T]` float waveform (32 kHz).

#     #     Returns
#     #     -------
#     #     embeddings : `torch.Tensor`
#     #         `[B, N, D]` patch-level embeddings.
#     #     """
#     #     if audio.ndim == 1:
#     #         audio = audio.unsqueeze(0)                                      # [B=1,T]
#     #     audio = audio.to(self.device)

#     #     audio = normalize_audio(audio)

#     #     # Process each audio sample independently
#     #     print(f"Audio shape before encoder: {audio.shape}")

#     #     # If audio is longer than 10 seconds, process it in chunks of 10 seconds
#     #     if audio.size(1) > self.custom_max_audio_length:
#     #         embeds = []
#     #         for chunk in audio.split(self.custom_max_audio_length, dim=-1):
#     #             # Process each chunk separately
#     #             chunk_embed = self._forward_10s_or_less(chunk)
#     #             embeds.append(chunk_embed)
#     #         embeddings = torch.cat(embeds, dim=1)
#     #     else:
#     #         embeddings = self._forward_10s_or_less(audio)

#     #     print(f"Embeddings shape after encoder: {embeddings.shape}")
#     #     return embeddings

#     # def _forward_10s_or_less(self, audio_batch: torch.Tensor) -> torch.Tensor:

#     #     audio_batch = pad(audio_batch, clip_len_sec=self.clip_length, sr=self.sampling_rate)    # pad to 10s

#     #     # log mel spectrogram  → [B, 1, n_mels, n_frames]
#     #     # mel only processes 1D tensors, so we need to compute the mel spectrogram independently for each element of the batch
#     #     specs = []
#     #     for i, wav in enumerate(audio_batch):
#     #         specs.append(self._apply_transforms(wav))

#     #     batch = torch.stack(specs)

#     #     embeddings = self.encoder(batch)
#     #     return embeddings

#     # def _apply_transforms(self, waveform: torch.Tensor) -> torch.Tensor:
#     #     """Apply transformation pipeline to a 1D waveform tensor."""

#     #     if waveform.ndim != 1 and waveform.ndim != 2:
#     #         raise ValueError(f"Expected 1D or 2D tensor, got {waveform.ndim}D tensor")
#     #     if waveform.ndim == 2:
#     #         waveform = waveform.mean(dim=0) # Down-mix stereo to mono

#     #     # Add channel dimension for fbank compatibility
#     #     wav = waveform.clone().unsqueeze(0)

#     #     # wav = wav.to(self.device)
#     #     spec = self.mel(wav)  # [1,128,256]

#     #     print(f"Spec shape after mel transform: {spec.shape}")

#     #     return spec
