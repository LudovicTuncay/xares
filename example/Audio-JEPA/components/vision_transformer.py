# A minimal ViT backbone – only the encoder path is kept.
import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .drop_path import DropPath
from flash_attn.modules.mha import MHA



# -----------------------------------------------------------------------------#
# Tiny helpers
# -----------------------------------------------------------------------------#
def _repeat_interleave_batch(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    """[B, N, D] → [B*repeat, N, D] by repeating each element along the batch dim."""
    if repeat == 1:
        return x
    return x.unsqueeze(1).repeat(1, repeat, 1, 1).flatten(0, 1)


def _apply_masks(x: torch.Tensor, masks):
    """No‑op placeholder – masking is only used during pre‑training."""
    return x
# -----------------------------------------------------------------------------#


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        use_flash_attn=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MHA(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, qkv_proj_bias=qkv_bias, use_flash_attn=use_flash_attn)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        patch_h, patch_w = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):                                  # [B,1,H,W] → [B,N,D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    """Backbone only (no [CLS] token, no predictor)."""

    def __init__(
        self,
        patch_size=(16, 16),
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_flash_attn=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        # Positional embedding – initialised for a 128 × 256 mel‑spec (→ 8 × 16 patches)
        n_init_patches = (128 // patch_size[1]) * (256 // patch_size[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, n_init_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # Transformer blocks
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                  drop_rate, attn_drop_rate, dpr[i], norm_layer=norm_layer, use_flash_attn=use_flash_attn)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, context_masks=None):                 # noqa: D401
        """
        Args
        ----
        x : [B, 1, n_mels, n_frames] float tensor in log‑mel space.
        Returns
        -------
        z : [B, N, D] where `N = floor(n_frames/patch_t) * floor(n_mels/patch_f)`
        """
        x = self.patch_embed(x)
        # interpolate / trim position embeddings if input size differs
        if x.size(1) != self.pos_embed.size(1):
            pos = self.pos_embed
            n = int(x.size(1) ** .5)
            pos = pos.reshape(1, int(math.sqrt(pos.size(1))), -1, self.embed_dim)\
                     .permute(0, 3, 1, 2)
            pos = nn.functional.interpolate(pos, size=(n, n), mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)
        else:
            pos = self.pos_embed
        x = x + pos

        if context_masks is not None:                       # ← kept for API completeness
            x = _apply_masks(x, context_masks)

        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)