
import os
import sys
import glob
import multiprocessing
import torch
import torch.nn as nn
import functools
import typing
import collections
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

# Hack to handle namespace collision for 'src' package
# We need to import 'src.models.audio_jepa_module' from audio-embeddings
# But xares also uses 'src' directory structure which might be needed for tasks.

# 1. Save state
orig_path = sys.path[:]
orig_src = sys.modules.get("src")

# 2. Add audio-embeddings to path
sys.path.insert(0, "/media/ltuncay/Shared-4TB/dev/audio-embeddings")

# 3. Force reload src for audio-embeddings
if "src" in sys.modules:
    del sys.modules["src"]

try:
    from src.models.audio_jepa_module import AudioJEPAModule
finally:
    # 4. Restore state
    sys.path = orig_path
    if orig_src:
        sys.modules["src"] = orig_src
    elif "src" in sys.modules:
        # If src was not present before, but is now (from audio-embeddings), define it.
        # BUT we want to allow xares to load its own src later. 
        # So we should delete it from sys.modules so imports resolve again using restored path.
        del sys.modules["src"]

# Note: Submodules 'src.models', 'src.models.audio_jepa_module' remain in sys.modules
# which is fine as xares doesn't have these specific submodules.

class AudioJEPAEncoder(nn.Module):
    def __init__(
        self, 
        output_dim: int = 768,
        sampling_rate: int = 16000,
        hop_size_in_ms: float = 10,
        max_audio_length_in_s: float = 10.0,  # Audio-JEPA was trained with 10s audio
        **kwargs
    ):
        super().__init__()
        self.output_dim = output_dim
        self.sampling_rate = sampling_rate
        self.hop_size_in_ms = hop_size_in_ms
        self.max_audio_length_in_s = max_audio_length_in_s
        
        checkpoint_path = os.environ.get("AUDIO_JEPA_CHECKPOINT")
        config_path = os.environ.get("AUDIO_JEPA_CONFIG")

        if not checkpoint_path or not config_path:
             raise ValueError("AUDIO_JEPA_CHECKPOINT and AUDIO_JEPA_CONFIG environment variables must be set.")
        
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
                            os.environ["AUDIO_JEPA_CHECKPOINT"] = os.path.abspath(checkpoint_path)
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
                    # If we are here, it means the env var update didn't propagate or wasn't done
                    raise RuntimeError(f"Ambiguous checkpoint path in worker process: {checkpoint_path}. Main process should have resolved this.")

        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        cfg = OmegaConf.load(config_path)
        
        # Load model from checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
            
        print(f"Loading AudioJEPAModule from {checkpoint_path} with config {config_path}")
        
        # Instantiate model. loading from checkpoint usually suffices.
        # Use safe_globals context manager to allow loading of specific types
        # We need to allow a few types that might be in the checkpoint (e.g. from Hydra config or partials)
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
        
        # Prepare init arguments that were excluded from hparams
        # The AudioJEPAModule expects 'net' and 'optimizer' in __init__
        init_args = {}
        if "model" in cfg and "net" in cfg.model:
            # Extract 'net' config from the Hydra config
            init_args["net"] = OmegaConf.to_container(cfg.model.net, resolve=True)
            # Create a dummy optimizer factory since we are in eval mode
            # AudioJEPAModule expects a callable that returns an optimizer
            init_args["optimizer"] = lambda params: torch.optim.AdamW(params)
        
        # Manual loading to handle meta-device initialization issues from timm
        # AudioJEPAModule.load_from_checkpoint fails if model inits on meta and weights aren't fully loaded/assigned before .to(cpu)
        
        # 1. Instantiate model
        print("Instantiating AudioJEPAModule...")
        with torch.serialization.safe_globals(safe_list):
             self.model = AudioJEPAModule(**init_args)

        # 2. Materialize meta parameters if any
        has_meta = False
        for param in self.model.parameters():
            if param.device.type == 'meta':
                has_meta = True
                break
        
        if has_meta:
            print("Detected meta parameters in AudioJEPAModule. Materializing to CPU...")
            self.model.to_empty(device='cpu')
        
        # 3. Load weights
        print(f"Loading weights from {checkpoint_path}")
        
        if checkpoint_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError:
                 raise ImportError("safetensors is required to load .safetensors files. Please install it with `uv pip install safetensors`.")
            
            checkpoint = load_file(checkpoint_path, device="cpu")
            # Safetensors usually contains just the state dict (flat), but let's check if it's nested (unlikely for safetensors)
            # Typically safetensors files are just the weight tensors.
            state_dict = checkpoint
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
        # 4. Apply weights
        # strict=False allows for missing keys if architecture slightly differs (e.g. loss or optimizer specific params)
        keys = self.model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
        
        self.model.eval()
        self.model.freeze()
        self.checkpoint_path = checkpoint_path

        # --- Fix 1: Calculate Effective Hop Size ---
        # Get spectrogram hop
        spec_config = init_args.get("net", {}).get("spectrogram", {})
        # Try ms first, then calculate from samples if needed (though config usually has ms)
        spec_hop_ms = spec_config.get("hop_length_ms", 10.0) 
        if "hop_length_ms" not in spec_config and "hop_length" in spec_config and "sample_rate" in spec_config:
             spec_hop_ms = (spec_config["hop_length"] / spec_config["sample_rate"]) * 1000.0

        # Get patch size (time dimension)
        # patch_embed config structure: net -> patch_embed -> patch_size
        # patch_size is usually [freq, time]
        patch_config = init_args.get("net", {}).get("patch_embed", {})
        patch_size = patch_config.get("patch_size", [16, 16])
        if isinstance(patch_size, (int, float)):
            patch_time_dim = patch_size
        else:
            patch_time_dim = patch_size[1] # Index 1 is time

        # Calculate effective hop
        # One token = patch_time_dim spectrogram frames
        self.hop_size_in_ms = spec_hop_ms * patch_time_dim
        print(f"Computed effective hop size: {self.hop_size_in_ms:.4f} ms (Spec Hop: {spec_hop_ms:.2f}ms * Patch Time: {patch_time_dim})")

        # --- Fix 2: Update output_dim from config ---
        net_config = init_args.get("net", {})
        if "encoder" in net_config and "embed_dim" in net_config["encoder"]:
            self.output_dim = net_config["encoder"]["embed_dim"]
            print(f"Updated output_dim to {self.output_dim} from config.")
        elif "patch_embed" in net_config and "embed_dim" in net_config["patch_embed"]:
            self.output_dim = net_config["patch_embed"]["embed_dim"]
            print(f"Updated output_dim to {self.output_dim} from config.")

        # Store frequency grid size for reshaping in forward
        # n_mels / patch_freq_dim
        n_mels = spec_config.get("n_mels", 128)
        patch_freq_dim = patch_size[0] if isinstance(patch_size, list) else patch_size
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
        # Audio input to xares encoders is typically [B, T]
        # AudioJEPAModule expects [B, 1, T] for waveform
        
        # Debug device (only prints once per instance ideally, but here every forward)
        # if not hasattr(self, "_device_checked"):
        #     print(f"AudioJEPAEncoder running on device: {self.model.device if hasattr(self.model, 'device') else next(self.parameters()).device}")
        #     self._device_checked = True
        
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        
        batch_size = audio.shape[0]
        num_samples = audio.shape[-1]
        chunk_samples = int(self.max_audio_length_in_s * self.sampling_rate)
        # Minimum 1 second to avoid kernel size issues
        min_chunk_samples = self.sampling_rate
        
        with torch.no_grad():
            # If audio fits in one chunk, process directly
            if num_samples <= chunk_samples:
                emb = self._process_chunk(audio) # Returns [B, T, D]
                # Create mask if needed (assuming all valid if no padding was done here, 
                # but input audio might be padded if batch_size > 1. 
                # However, without attention_mask passed in, we assume full audio is valid?
                # If attention_mask is passed, we should downsample it.
                # For simplicity in this fix, we generate mask based on output shape (all ones).
                # To be proper, we should respect input attention_mask.
                # But let's start with returning a mask of ones for the output length.
                
                # If we want to support input attention_mask, we need to map sample index to token index.
                # hop_size_in_ms tells us duration per token.
                # token_duration_samples = hop_size_in_ms * sample_rate / 1000
                
                # mask = torch.ones(emb.shape[0], emb.shape[1], device=emb.device, dtype=torch.long)
                
                # If attention_mask was provided for audio, we should technically use it.
                # But AudioJEPA is a windowed model, so exact masking is tricky.
                # Let's rely on the fact that we handle padding below if we chunk.
                # Here we assume audio was roughly same length or we treat padding as valid audio for now.
                
                # If we return None, xares might be unhappy if it expects a mask.
                # The README says "outputs (features, feature_attention_mask)".
                
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
            # attention_masks = []
            
            for emb in batch_embeddings:
                current_len = emb.shape[0]
                # mask = torch.ones(current_len, device=emb.device, dtype=torch.long)
                
                if current_len < max_len:
                    pad = torch.zeros(max_len - current_len, emb.shape[1], device=emb.device, dtype=emb.dtype)
                    emb = torch.cat([emb, pad], dim=0)
                    
                    # mask_pad = torch.zeros(max_len - current_len, device=emb.device, dtype=torch.long)
                    # mask = torch.cat([mask, mask_pad], dim=0)
                    
                padded_embeddings.append(emb)
                # attention_masks.append(mask)
            
            return torch.stack(padded_embeddings, dim=0)
