from __future__ import annotations

import collections
import functools
import glob
import multiprocessing
import os
import sys
import typing

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

AUDIO_EMBEDDINGS_ROOT = os.environ.get(
    "AUDIO_EMBEDDINGS_ROOT",
    "/media/ltuncay/Shared-4TB/dev/audio-embeddings",
)

orig_path = sys.path[:]
orig_src = sys.modules.get("src")
sys.path.insert(0, AUDIO_EMBEDDINGS_ROOT)

if "src" in sys.modules:
    del sys.modules["src"]

try:
    from src.models.best_rq3_module import BestRQ3Module
finally:
    sys.path = orig_path
    if orig_src:
        sys.modules["src"] = orig_src
    elif "src" in sys.modules:
        del sys.modules["src"]


MODEL_INIT_ARGS = {
    "warmup_pct",
    "final_lr_ratio",
    "codebook_dim",
    "vocab_size",
    "z_loss_weight",
    "ema",
    "compile",
}


def _to_plain_config(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    if not os.path.isdir(checkpoint_path):
        return checkpoint_path

    ckpt_files = sorted(
        glob.glob(os.path.join(checkpoint_path, "*.ckpt"))
        + glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    )
    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt or .safetensors files found in directory: {checkpoint_path}"
        )

    if len(ckpt_files) == 1:
        return ckpt_files[0]

    if multiprocessing.parent_process() is not None:
        raise RuntimeError(
            f"Ambiguous checkpoint path in worker process: {checkpoint_path}. "
            "Run tools/run_best_rq3_eval.py so the main process resolves it first."
        )

    print(f"\nMultiple checkpoints found in {checkpoint_path}:")
    default_idx = -1
    for idx, path in enumerate(ckpt_files):
        name = os.path.basename(path)
        if name in {"last.ckpt", "last.safetensors"}:
            default_idx = idx
            print(f"[{idx}] {name} (default)")
        else:
            print(f"[{idx}] {name}")

    prompt = "Select checkpoint index"
    if default_idx != -1:
        prompt += f" [default: {default_idx}]"
    prompt += ": "

    try:
        selection = input(prompt)
        if not selection.strip() and default_idx != -1:
            selection = str(default_idx)
        selected_idx = int(selection)
        if not 0 <= selected_idx < len(ckpt_files):
            raise ValueError(f"Index {selected_idx} out of range")
    except KeyboardInterrupt:
        print("\nSelection cancelled by user.")
        sys.exit(1)
    except ValueError as exc:
        raise ValueError(f"Invalid selection: {exc}") from exc

    checkpoint_path = os.path.abspath(ckpt_files[selected_idx])
    os.environ["BEST_RQ3_CHECKPOINT"] = checkpoint_path
    print(f"Selected: {checkpoint_path}\n")
    return checkpoint_path


def _load_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install it with `uv pip install safetensors`."
            ) from exc
        return load_file(checkpoint_path, device="cpu")

    safe_list = [
        functools.partial,
        torch.optim.AdamW,
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        typing.Any,
        dict,
        list,
        set,
        tuple,
        int,
        float,
        str,
        bool,
        collections.defaultdict,
        collections.OrderedDict,
        AnyNode,
        torch.nn.MSELoss,
        torch.nn.CrossEntropyLoss,
    ]
    with torch.serialization.safe_globals(safe_list):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _filter_compatible_state_dict(model: nn.Module, state_dict: dict) -> dict:
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        expected = model_state.get(key)
        if expected is None:
            continue
        if expected.shape != value.shape:
            skipped.append((key, tuple(value.shape), tuple(expected.shape)))
            continue
        compatible[key] = value

    if skipped:
        print(f"Skipped {len(skipped)} checkpoint tensors with incompatible shapes.")
        for key, found_shape, expected_shape in skipped[:10]:
            print(f"  {key}: checkpoint {found_shape} != model {expected_shape}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    return compatible


def _frontend_receptive_params(
    conv_layers_spec: list[tuple[int, int, int]],
) -> tuple[int, int]:
    stride_total = 1
    receptive_field = 1
    for _, kernel, stride in conv_layers_spec:
        receptive_field += (int(kernel) - 1) * stride_total
        stride_total *= int(stride)
    return receptive_field, stride_total


class BestRQ3Encoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 768,
        sampling_rate: int = 16000,
        hop_size_in_ms: float = 20.0,
        max_audio_length_in_s: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        checkpoint_path = os.environ.get("BEST_RQ3_CHECKPOINT")
        config_path = os.environ.get("BEST_RQ3_CONFIG")
        if not checkpoint_path or not config_path:
            raise ValueError(
                "BEST_RQ3_CHECKPOINT and BEST_RQ3_CONFIG environment variables "
                "must be set."
            )

        checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        cfg = OmegaConf.load(config_path)
        model_cfg = cfg.get("model", None)
        model_target = None
        if model_cfg is not None:
            model_target = str(model_cfg.get("_target_", ""))
        if model_target != "src.models.best_rq3_module.BestRQ3Module":
            raise ValueError(
                "BestRQ3Encoder only supports "
                "'src.models.best_rq3_module.BestRQ3Module', got "
                f"'{model_target}'."
            )

        init_args = {
            "net": OmegaConf.to_container(cfg.model.net, resolve=True),
            "optimizer": lambda params: torch.optim.AdamW(params),
        }
        for key in MODEL_INIT_ARGS:
            if key in cfg.model:
                init_args[key] = _to_plain_config(cfg.model[key])

        self.output_dim = int(
            init_args.get("net", {}).get("encoder", {}).get("embed_dim", output_dim)
        )

        sampling_config = init_args.get("net", {}).get("sampling", {})
        self.sampling_rate = int(sampling_config.get("sample_rate", sampling_rate))

        print(
            "Loading BEST-RQ-3 model from "
            f"{checkpoint_path} with config {config_path}"
        )
        self.model = BestRQ3Module(**init_args)

        has_meta = any(
            param.device.type == "meta" for param in self.model.parameters()
        )
        if has_meta:
            print("Detected meta parameters. Materializing to CPU...")
            self.model.to_empty(device="cpu")

        state_dict = _load_state_dict(checkpoint_path)
        state_dict = _filter_compatible_state_dict(self.model, state_dict)
        keys = self.model.load_state_dict(state_dict, strict=False)
        print(
            "Weights loaded. "
            f"Missing keys: {len(keys.missing_keys)}, "
            f"Unexpected keys: {len(keys.unexpected_keys)}"
        )

        self.model.eval()
        self.model.freeze()
        self.checkpoint_path = checkpoint_path

        conv_layers_spec = self.model.feature_encoder.conv_layers_spec
        receptive_field, stride = _frontend_receptive_params(conv_layers_spec)
        self._min_chunk_samples = max(self.sampling_rate, receptive_field)
        self._feature_stride_samples = stride
        self.hop_size_in_ms = (stride / self.sampling_rate) * 1000.0

        if max_audio_length_in_s is None:
            max_audio_length_in_s = float(sampling_config.get("section_seconds", 10.0))
        self.max_audio_length_in_s = float(max_audio_length_in_s)

        print(
            "Computed BEST-RQ-3 hop size: "
            f"{self.hop_size_in_ms:.4f} ms "
            f"(CNN stride: {stride} samples at {self.sampling_rate} Hz)"
        )

    def _valid_token_count(self, num_samples: int, padded_num_samples: int) -> int:
        total_tokens = self.model.feature_encoder.total_patches(padded_num_samples)
        valid_tokens = self.model.feature_encoder.total_patches(num_samples)
        if valid_tokens <= 0:
            return max(1, min(1, total_tokens))
        return min(valid_tokens, total_tokens)

    def _process_chunk(
        self,
        chunk: torch.Tensor,
        valid_samples: int | None = None,
    ) -> torch.Tensor:
        emb = self.model(chunk)
        if emb.ndim != 3:
            raise RuntimeError(
                f"Expected BEST-RQ-3 encoder output [B, T, D], got {tuple(emb.shape)}"
            )
        if valid_samples is None or valid_samples == chunk.shape[-1]:
            return emb
        valid_tokens = self._valid_token_count(valid_samples, chunk.shape[-1])
        return emb[:, :valid_tokens, :]

    @torch.inference_mode()
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise ValueError(
                f"Expected audio with shape [B, T] or [T], got {tuple(audio.shape)}"
            )

        batch_size, num_samples = audio.shape
        chunk_samples = int(self.max_audio_length_in_s * self.sampling_rate)
        chunk_samples = max(chunk_samples, self._min_chunk_samples)

        if num_samples <= chunk_samples:
            if num_samples < self._min_chunk_samples:
                padded = torch.zeros(
                    batch_size,
                    self._min_chunk_samples,
                    device=audio.device,
                    dtype=audio.dtype,
                )
                padded[:, :num_samples] = audio
                return self._process_chunk(padded, valid_samples=num_samples)
            return self._process_chunk(audio)

        batch_embeddings = []
        for batch_idx in range(batch_size):
            single_audio = audio[batch_idx : batch_idx + 1]
            sample_chunks = []
            for start in range(0, num_samples, chunk_samples):
                end = min(start + chunk_samples, num_samples)
                chunk = single_audio[:, start:end]
                valid_samples = chunk.shape[-1]

                if valid_samples < self._min_chunk_samples:
                    padded = torch.zeros(
                        1,
                        self._min_chunk_samples,
                        device=chunk.device,
                        dtype=chunk.dtype,
                    )
                    padded[:, :valid_samples] = chunk
                    chunk = padded

                sample_chunks.append(
                    self._process_chunk(chunk, valid_samples=valid_samples).squeeze(0)
                )

            batch_embeddings.append(torch.cat(sample_chunks, dim=0))

        max_len = max(emb.shape[0] for emb in batch_embeddings)
        padded_embeddings = []
        for emb in batch_embeddings:
            if emb.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - emb.shape[0],
                    emb.shape[1],
                    device=emb.device,
                    dtype=emb.dtype,
                )
                emb = torch.cat([emb, pad], dim=0)
            padded_embeddings.append(emb)

        return torch.stack(padded_embeddings, dim=0)
