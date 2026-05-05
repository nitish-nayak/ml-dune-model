import torch
import torch.nn as nn
from typing import Literal, Optional
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.nn.modules.base_module import BaseSpatialModule
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.geometry.features.ops.convert import cat_to_pad_tensor

from warpconvnet.nn.modules.attention import offset_to_mask, zero_out_points
from warpconvnet.nn.modules.attention import Attention, ToSpatialFeatures

try:
    import flash_attn
except ImportError:
    flash_attn = None

### This is based on what offered by WarpConvNet but adjusting
### the final tensor shape of the spatial encoding
### to support both standard and flash attention mechanism.
class ToAttentionSmart(BaseSpatialModule):
    def __init__(
        self,
        out_channels: int,
        use_encoding: bool = False,
        num_encoding_channels: Optional[int] = None,
        encoding_range: Optional[float] = None,
        num_heads: int = 1,
        concat_input: bool = True,
        num_spatial_features: int = 3,
        out_type: Literal["nested", "cat"] = "cat",
        # NEW: how wide should pos_enc be?
        # - "per_head" -> out_channels // num_heads  (for non-flash: add to Q/K)
        # - "full"     -> out_channels              (for flash: add to x)
        pos_enc_mode: Literal["per_head", "full"] = "per_head",
    ):
        super().__init__()
        self.out_type = out_type
        self.use_encoding = use_encoding

        if use_encoding:
            assert num_encoding_channels is not None, "num_encoding_channels must be provided"
            assert encoding_range is not None, "encoding_range must be provided"
            assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

            if pos_enc_mode == "per_head":
                pos_out = out_channels // num_heads
            elif pos_enc_mode == "full":
                pos_out = out_channels
            else:
                raise ValueError(f"Unknown pos_enc_mode={pos_enc_mode}")

            in_feats = num_encoding_channels * num_spatial_features + (
                num_spatial_features if concat_input else 0
            )

            print("SinusouidalEncoding settings:")
            print(f"  num_channels={num_encoding_channels}")
            print(f"  data_range={encoding_range}")
            print(f"  concat_input={concat_input}")

            self.encoding = nn.Sequential(
                SinusoidalEncoding(
                    num_channels=num_encoding_channels,
                    data_range=encoding_range,
                    concat_input=concat_input,
                ),
                nn.Linear(in_feats, pos_out),
            )

    def forward(self, x: Geometry):
        if self.out_type == "nested":
            features = x.nested_features
            coordinates = x.nested_coordinates
            # NOTE: if nested path is used, you'll need offsets for padding/mask;
            # leaving as-is since your current usage appears out_type="cat".
        else:
            features_cat, offsets = x.features, x.offsets
            features = cat_to_pad_tensor(features_cat, offsets)          # [B, N, C]
            coordinates = x.coordinate_tensor                            # [M, D]
            num_points = offsets.diff()                                  # [B]

        if self.use_encoding:
            pos_enc_cat = self.encoding(coordinates)                     # [M, pos_out]
            pos_enc = cat_to_pad_tensor(pos_enc_cat, offsets)            # [B, N, pos_out]
        else:
            pos_enc = None
            pos_enc_cat = None

        mask = offset_to_mask(features, offsets, features.shape[1])      # [B, 1, N, N] (bool)
        return features, pos_enc, pos_enc_cat, mask, num_points


### This is based on what offered by WarpConvNet but adjusting
### the expected spatial dimensions to 2D. It also uses the
### ToAttentionSmart() block to support both standard and flash attention mechanism
### in case spatial encoding is enabled.
class SpatialFeatureAttention2D(Attention):
    """
    SpatialFeatureAttention for 2D coordinates (x, y).
    Supports:
      - flash ON/OFF
      - encoding ON/OFF
    and chooses positional encoding width to be compatible with the injection site.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_encoding_channels: int = 32,
        encoding_range: float = 1.0,
        use_encoding: bool = True,
        enable_flash: bool = True,
        use_batched_qkv: bool = True,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            use_batched_qkv=use_batched_qkv,
        )

        # Decide how wide pos_enc should be:
        # - flash path adds pos_enc to x -> needs full C
        # - non-flash adds pos_enc to q/k per head -> head_dim
        pos_enc_mode = "full" if (enable_flash and use_encoding) else "per_head"

        self.to_attn = ToAttentionSmart(
            out_channels=dim,
            use_encoding=use_encoding,
            num_encoding_channels=num_encoding_channels,
            encoding_range=encoding_range,
            num_heads=num_heads,
            concat_input=True,
            num_spatial_features=2,     # <-- the whole point: 2D
            out_type="cat",
            pos_enc_mode=pos_enc_mode,  # <-- new: resolves flash/encoding mismatch
        )
        self.from_attn = ToSpatialFeatures()

    def forward(self, x: Geometry) -> Geometry:

        # extract padded tensors from the original concatenated tensor
        # this is used for the non flash path with an appropriate mask
        features, pos_enc, pos_enc_cat, mask, num_points = self.to_attn(x)

        if not self.enable_flash:
            B, N, C = features.shape
            qkv = self.qkv(features).reshape(B, N, 3, C)

            # Reshape to [B, N, 3, num_heads, head_dim]
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)

            # Note: with flash + encoding, pos_enc is [B, N, C] so x + pos_enc works.
            # With non-flash + encoding, pos_enc is [B, N, head_dim] so q/k addition works.
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            # Apply positional encoding to the query and key (non-flash path)
            if pos_enc is not None:
                q = q + pos_enc.unsqueeze(1)
                k = k + pos_enc.unsqueeze(1)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                assert mask.device == attn.device
                attn_bias = torch.zeros(mask.shape, dtype = attn.dtype, device = attn.device)
                attn_bias.masked_fill_(mask.logical_not(), float("-1e9"))
                attn = attn + attn_bias

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            y = attn @ v
            y = y.transpose(1, 2).reshape(B, N, C)

            y = self.proj(y)
            y = self.proj_drop(y)

            if num_points is not None:
                y = zero_out_points(y, num_points)

            return self.from_attn(y, x)
        # use flash_attn on the concatenated tensor directly, no need to use padded and convert it back
        else:
            feats, offsets = x.features, x.offsets
            M, C = feats.shape[:2]
            if pos_enc_cat is not None:
                pos_enc_cat_scaled = pos_enc_cat * 0.05
                #  print("Comparing norms, feat vs pos_enc : ", feats.norm(p=2), pos_enc_cat_scaled.norm(p=2))
                #  print("Comparing shapes, feat vs pos_enc : ", feats.shape, pos_enc_cat_scaled.shape)
                feats = feats + pos_enc_cat_scaled

            qkv = self.qkv(feats).reshape(M, 3, self.num_heads, C // self.num_heads)
            if qkv.dtype not in [torch.float16, torch.bfloat16]:
                qkv = qkv.to(torch.float16)
            # Warning: When the loss is NaN, this module will fail during backward with
            # index out of bounds error.
            # e.g. /pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:144: operator(): block: [192,0,0], thread: [32,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "
            # https://discuss.pytorch.org/t/scattergatherkernel-cu-assertion-idx-dim-0-idx-dim-index-size-index-out-of-bounds/195356
            max_seqlen = int(offsets.diff().max())
            attn_offsets = offsets.to(device=qkv.device, dtype=torch.int32)
            out_feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv,
                attn_offsets,
                max_seqlen=max_seqlen,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
            )
            out_feat = out_feat.reshape(M, C).to(feats.dtype)

            out_feat = self.proj(out_feat)
            out_feat = self.proj_drop(out_feat)

            return x.replace(batched_features=out_feat.to(feats.dtype))
        return
