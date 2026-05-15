import os
import sys
import glob
import multiprocessing
import tempfile
import torch
import torch.nn as nn
import functools
import typing
import collections
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

AUDIO_EMBEDDINGS_ROOT = "/media/ltuncay/Shared-4TB/dev/audio-embeddings"

# Hack to handle namespace collision for 'src' package
orig_path = sys.path[:]
orig_src = sys.modules.get("src")

# Add audio-embeddings to path
sys.path.insert(0, AUDIO_EMBEDDINGS_ROOT)

if "src" in sys.modules:
    del sys.modules["src"]

try:
    from src.models.best_rq_module import BestRQModule
    from src.models.best_rq2_module import BestRQ2Module
finally:
    sys.path = orig_path
    if orig_src:
        sys.modules["src"] = orig_src
    elif "src" in sys.modules:
        del sys.modules["src"]

BEST_RQ_MODULES = {
    "src.models.best_rq_module.BestRQModule": BestRQModule,
    "src.models.best_rq2_module.BestRQ2Module": BestRQ2Module,
}

MODEL_SHAPE_AND_EVAL_ARGS = {
    "spectrogram_adjustment_mode",
    "codebook_dim",
    "vocab_size",
    "quantizer",
    "rvq",
}

def _to_plain_config(value, *, resolve: bool = True):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=resolve)
    return value


def _is_enabled(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _infer_quantizer_input_dim(net_config: dict) -> int:
    patch_config = net_config.get("patch_embed", {})
    patch_size = patch_config.get("patch_size", [16, 16])
    if isinstance(patch_size, (int, float)):
        patch_h = patch_w = int(patch_size)
    else:
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])
    in_chans = int(patch_config.get("in_chans", 1))
    return patch_h * patch_w * in_chans


def _save_temp_artifact(artifact: dict) -> str:
    fd, artifact_path = tempfile.mkstemp(prefix="xares_best_rq_quantizer_", suffix=".pt")
    os.close(fd)
    torch.save(artifact, artifact_path)
    return artifact_path


def _make_flat_dummy_quantizer_artifact(
    *,
    input_dim: int,
    codebook_dim: int,
    vocab_size: int,
    quantizer_type: str,
) -> str:
    projection = torch.empty(input_dim, codebook_dim)
    nn.init.xavier_uniform_(projection)

    if quantizer_type == "stratified":
        artifact = {
            "projection": projection,
            "energy_thresholds": torch.empty(0),
            "code_counts": torch.tensor([vocab_size], dtype=torch.long),
            "centroids_per_bin": [torch.randn(vocab_size, codebook_dim)],
        }
    else:
        artifact = {
            "projection": projection,
            "centroids": torch.randn(vocab_size, codebook_dim),
        }
    return _save_temp_artifact(artifact)


def _make_rvq_dummy_quantizer_artifact(
    *,
    input_dim: int,
    codebook_dim: int,
    num_stages: int,
    stage_vocab_size: int,
) -> str:
    projection = torch.empty(input_dim, codebook_dim)
    nn.init.xavier_uniform_(projection)
    return _save_temp_artifact(
        {
            "projection": projection,
            "stage_codebooks": torch.randn(num_stages, stage_vocab_size, codebook_dim),
        }
    )


def _make_quantizers_eval_safe(init_args: dict) -> list[str]:
    """Use temporary target quantizer artifacts that encoder-only eval never reads."""
    temp_artifacts = []
    if not _is_enabled(os.environ.get("BEST_RQ_SKIP_TARGET_QUANTIZER")):
        return temp_artifacts

    net_config = init_args.get("net", {})
    quantizer_input_dim = _infer_quantizer_input_dim(net_config)
    codebook_dim = int(init_args.get("codebook_dim", 16))
    vocab_size = int(init_args.get("vocab_size", 8192))

    quantizer = init_args.get("quantizer")
    if isinstance(quantizer, dict) and quantizer.get("type") != "random":
        quantizer = dict(quantizer)
        quantizer_type = str(quantizer.get("type", "fitted"))
        artifact_path = quantizer.get("artifact_path")
        dummy_path = _make_flat_dummy_quantizer_artifact(
            input_dim=quantizer_input_dim,
            codebook_dim=codebook_dim,
            vocab_size=vocab_size,
            quantizer_type=quantizer_type,
        )
        quantizer["artifact_path"] = dummy_path
        init_args["quantizer"] = quantizer
        temp_artifacts.append(dummy_path)
        print(
            "Using temporary Best-RQ target quantizer artifact for encoder eval"
            + (f": {artifact_path}" if artifact_path else "")
        )

    rvq = init_args.get("rvq")
    if isinstance(rvq, dict) and rvq.get("enabled", False) and rvq.get("mode") != "random":
        artifact_path = rvq.get("artifact_path")
        rvq = dict(rvq)
        dummy_path = _make_rvq_dummy_quantizer_artifact(
            input_dim=quantizer_input_dim,
            codebook_dim=codebook_dim,
            num_stages=int(rvq.get("num_stages", 4)),
            stage_vocab_size=int(rvq.get("stage_vocab_size", 512)),
        )
        rvq["artifact_path"] = dummy_path
        init_args["rvq"] = rvq
        temp_artifacts.append(dummy_path)
        print(
            "Using temporary Best-RQ RVQ target quantizer artifact for encoder eval"
            + (f": {artifact_path}" if artifact_path else "")
        )

    return temp_artifacts


def _cleanup_temp_artifacts(paths: list[str]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


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


class BestRQEncoder(nn.Module):
    def __init__(
        self, 
        output_dim: int = 768,
        sampling_rate: int = 16000,
        hop_size_in_ms: float = 10,
        max_audio_length_in_s: float = 10.0,
        **kwargs
    ):
        super().__init__()
        self.output_dim = output_dim
        self.sampling_rate = sampling_rate
        self.hop_size_in_ms = hop_size_in_ms
        self.max_audio_length_in_s = max_audio_length_in_s
        
        checkpoint_path = os.environ.get("BEST_RQ_CHECKPOINT")
        config_path = os.environ.get("BEST_RQ_CONFIG")

        if not checkpoint_path or not config_path:
             raise ValueError("BEST_RQ_CHECKPOINT and BEST_RQ_CONFIG environment variables must be set.")
        
        if os.path.isdir(checkpoint_path):
            # Find all .ckpt and .safetensors files
            ckpt_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.ckpt")) + glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
            if not ckpt_files:
                raise FileNotFoundError(f"No .ckpt or .safetensors files found in directory: {checkpoint_path}")
            
            if len(ckpt_files) == 1:
                checkpoint_path = ckpt_files[0]
            else:
                # Multiple files found: ask user if main process
                if multiprocessing.parent_process() is None:
                    print(f"\nMultiple checkpoints found in {checkpoint_path}:")
                    default_idx = -1
                    for idx, f in enumerate(ckpt_files):
                        fname = os.path.basename(f)
                        if fname in ["last.ckpt", "last.safetensors"]:
                            default_idx = idx
                            print(f"[{idx}] {fname} (default)")
                        else:
                            print(f"[{idx}] {fname}")
                    
                    prompt = "Select checkpoint index"
                    if default_idx != -1:
                        prompt += f" [default: {default_idx}]"
                    prompt += ": "
                    
                    try:
                        selection = input(prompt)
                        if not selection.strip() and default_idx != -1:
                            selection = str(default_idx)
                        
                        selected_idx = int(selection)
                        if 0 <= selected_idx < len(ckpt_files):
                            checkpoint_path = ckpt_files[selected_idx]
                            # CRITICAL: Update env var so workers inherit it
                            os.environ["BEST_RQ_CHECKPOINT"] = os.path.abspath(checkpoint_path)
                            print(f"Selected: {checkpoint_path}\n")
                        else:
                            raise ValueError(f"Index {selected_idx} out of range")
                    except KeyboardInterrupt:
                        print("\nSelection cancelled by user.")
                        sys.exit(1)
                    except ValueError as e:
                        raise ValueError(f"Invalid selection: {e}")
                else:
                    # Worker process should have received a specific file path via inherited env var
                    raise RuntimeError(f"Ambiguous checkpoint path in worker process: {checkpoint_path}. Main process should have resolved this.")

        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        cfg = OmegaConf.load(config_path)
        
        # Load model from checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
            
        print(f"Loading Best-RQ model from {checkpoint_path} with config {config_path}")
        
        # Instantiate model. loading from checkpoint usually suffices.
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
        
        init_args = {}
        model_cfg = cfg.get("model", None)
        model_target = "src.models.best_rq_module.BestRQModule"
        if model_cfg is not None:
            model_target = str(model_cfg.get("_target_", model_target))

        model_cls = BEST_RQ_MODULES.get(model_target)
        if model_cls is None:
            raise ValueError(
                f"Unsupported Best-RQ model target '{model_target}'. "
                f"Supported targets: {sorted(BEST_RQ_MODULES)}"
            )

        if "model" in cfg and "net" in cfg.model:
            init_args["net"] = OmegaConf.to_container(cfg.model.net, resolve=True)
            init_args["optimizer"] = lambda params: torch.optim.AdamW(params)

            for key in MODEL_SHAPE_AND_EVAL_ARGS:
                if key in cfg.model:
                    init_args[key] = _to_plain_config(
                        cfg.model[key],
                        resolve=key not in {"quantizer", "rvq"},
                    )
        elif "net" in cfg:
             init_args["net"] = OmegaConf.to_container(cfg.net, resolve=True)
             init_args["optimizer"] = lambda params: torch.optim.AdamW(params)

        # The checkpoint config determines the embedding dimension. Set this before
        # loading so downstream errors report the actual model dimensions.
        encoder_config = init_args.get("net", {}).get("encoder", {})
        if "embed_dim" in encoder_config:
            self.output_dim = encoder_config["embed_dim"]

        temp_quantizer_artifacts = _make_quantizers_eval_safe(init_args)
        
        # 1. Instantiate model
        print(f"Instantiating {model_cls.__name__}...")
        try:
            with torch.serialization.safe_globals(safe_list):
                 self.model = model_cls(**init_args)
        finally:
            _cleanup_temp_artifacts(temp_quantizer_artifacts)

        # 2. Materialize meta parameters if any
        has_meta = False
        for param in self.model.parameters():
            if param.device.type == 'meta':
                has_meta = True
                break
        
        if has_meta:
            print("Detected meta parameters. Materializing to CPU...")
            self.model.to_empty(device='cpu')
        
        # 3. Load weights
        print(f"Loading weights from {checkpoint_path}")
        
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError:
                 raise ImportError("safetensors is required to load .safetensors files. Please install it with `uv pip install safetensors`.")
            
            checkpoint = load_file(checkpoint_path, device="cpu")
            state_dict = checkpoint
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

        state_dict = _filter_compatible_state_dict(self.model, state_dict)
            
        # 4. Apply weights
        keys = self.model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
        
        self.model.eval()
        self.model.freeze()
        self.checkpoint_path = checkpoint_path

        # Set output_dim from encoder config if available
        encoder_config = init_args.get("net", {}).get("encoder", {})
        if "embed_dim" in encoder_config:
            self.output_dim = encoder_config["embed_dim"]

        # --- Calculate Effective Hop Size ---
        # Get spectrogram hop
        spec_config = init_args.get("net", {}).get("spectrogram", {})
        spec_hop_ms = spec_config.get("hop_length_ms", 10.0) 
        if "hop_length_ms" not in spec_config and "hop_length" in spec_config and "sample_rate" in spec_config:
             spec_hop_ms = (spec_config["hop_length"] / spec_config["sample_rate"]) * 1000.0

        # Get patch size (time dimension)
        patch_config = init_args.get("net", {}).get("patch_embed", {})
        patch_size = patch_config.get("patch_size", [16, 16])
        if isinstance(patch_size, (int, float)):
            patch_time_dim = patch_size
            patch_freq_dim = patch_size
        else:
            patch_time_dim = patch_size[1] # Index 1 is time
            patch_freq_dim = patch_size[0]

        # Calculate effective hop
        self.hop_size_in_ms = spec_hop_ms * patch_time_dim
        print(f"Computed effective hop size: {self.hop_size_in_ms:.4f} ms (Spec Hop: {spec_hop_ms:.2f}ms * Patch Time: {patch_time_dim})")

        # Store frequency grid size for reshaping in forward
        n_mels = spec_config.get("n_mels", 128)
        self.freq_grid_size = n_mels // patch_freq_dim
        
    def _process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Process a single chunk and reshape output to be time-major.
        Args:
            chunk: [B, 1, T] usually
        Returns:
            emb: [B, Time, D]
        """
        # Model output is [B, N, D] where N = F_grid * T_grid
        emb = self.model(chunk) # [B, N, D]
        
        B, N, D = emb.shape
        F_grid = int(self.freq_grid_size)
        T_grid = N // F_grid
        
        emb = emb.view(B, F_grid, T_grid, D)
        
        # Average over frequency dimension to get single time sequence
        emb = emb.mean(dim=1) # [B, T_grid, D]
        
        return emb

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        
        batch_size = audio.shape[0]
        num_samples = audio.shape[-1]
        chunk_samples = int(self.max_audio_length_in_s * self.sampling_rate)
        min_chunk_samples = self.sampling_rate
        
        with torch.no_grad():
            # If audio fits in one chunk, process directly
            if num_samples <= chunk_samples:
                emb = self._process_chunk(audio) # Returns [B, T, D]
                # mask = torch.ones(emb.shape[0], emb.shape[1], device=emb.device, dtype=torch.long)
                return emb
            
            # Process each sample in the batch with chunking
            batch_embeddings = []
            for b in range(batch_size):
                single_audio = audio[b:b+1]  # Keep batch dim [1, 1, T]
                sample_chunks = []
                
                for start in range(0, num_samples, chunk_samples):
                    end = min(start + chunk_samples, num_samples)
                    remaining = end - start
                    
                    # Handle short tail chunks by padding
                    if remaining < min_chunk_samples:
                        chunk = single_audio[..., start:end]
                        padded = torch.zeros(1, 1, min_chunk_samples, device=chunk.device, dtype=chunk.dtype)
                        padded[..., :remaining] = chunk
                        chunk_emb = self._process_chunk(padded).squeeze(0)  # [1, T', D] -> [T', D]
                        # Keep only valid frames (proportional to actual length)
                        valid_frames = max(1, int(chunk_emb.shape[0] * remaining / min_chunk_samples))
                        sample_chunks.append(chunk_emb[:valid_frames])
                    else:
                        chunk = single_audio[..., start:end]
                        chunk_emb = self._process_chunk(chunk).squeeze(0)  # [1, T', D] -> [T', D]
                        sample_chunks.append(chunk_emb)
                
                # Concatenate all chunks along time dimension
                full_emb = torch.cat(sample_chunks, dim=0) if len(sample_chunks) > 1 else sample_chunks[0]
                batch_embeddings.append(full_emb)
            
            # Pad to same length and stack
            max_len = max(emb.shape[0] for emb in batch_embeddings)
            padded_embeddings = []
            attention_masks = []
            
            for emb in batch_embeddings:
                current_len = emb.shape[0]
                mask = torch.ones(current_len, device=emb.device, dtype=torch.long)
                
                if current_len < max_len:
                    pad = torch.zeros(max_len - current_len, emb.shape[1], device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=0)
                    
                    mask_pad = torch.zeros(max_len - current_len, device=emb.device, dtype=torch.long)
                    mask = torch.cat([mask, mask_pad], dim=0)
                    
                padded_embeddings.append(emb)
                attention_masks.append(mask)
            
            return torch.stack(padded_embeddings, dim=0)
