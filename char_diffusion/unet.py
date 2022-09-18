from typing import *
from jaxtyping import Float, Array

import math
import jax
import jax.numpy as jnp
import numpy as np

import equinox.nn as nn
from equinox import Module, static_field

from functools import partial
from einops import rearrange

from .custom_layers import Conv1d, GroupNorm, Linear


DType = NewType("DType", jax.numpy.dtype)
Shape = NewType("Shape", Tuple[int, ...])
PRNGKey = NewType("PRNGKey", jax._src.prng.PRNGKeyArray)


class SiLU(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Array, key: Optional[PRNGKey] = None) -> Array:
        return jax.nn.silu(x)


# Attention


def multihead_attn(
    q: Array, k: Array, v: Array, scale: float, bias: Array = jnp.array([0.0])
) -> Array:
    """Scaled Dot-Product Attention ("Soft Look-Up Table")."""
    # Computes similarity scores ("attention logits"): [..., head_dim, seq_len, seq_len]
    score = jnp.einsum("... h s d, ... h S d -> ... h s S", q, k)
    score = score * scale + bias  # Normalized similarity scores
    # NOTE: Similarity distribution has peaks where query most matches a key.
    weight = jax.nn.softmax(score)
    # Attention "look-up table".
    attn = jnp.einsum("... h s S, ... h S d -> ... h s d", weight, v)
    return attn


def split_fused_proj(proj: Array, dims: Shape) -> Tuple[Array, ...]:
    """Splits fused input projection tensor (e.g. when you use 1 dense matrix to compute q,k,v)."""
    indices = np.cumsum(dims)
    return jnp.split(proj, indices[:-1], axis=-1)


def split_heads(x: Array, num_heads: int) -> Array:
    """Splits the `n` input heads."""
    # s = seq_len, h = num_heads, d = head_dim
    return rearrange(x, "... s (h d) -> ... h s d", h=num_heads)


def merge_heads(x: Array) -> Array:
    """Concatenates the input multi-heads (reverse of `split_heads`)."""
    # s = seq_len, h = num_heads, d = head_dim, (h d) = h * d = embed_dim
    return rearrange(x, "... h s d -> ... s (h d)")  # "concat"


class SelfAttentionBlock(Module):
    """
    Reference: 
    - https://github.com/nshepperd/jax-guided-diffusion/blob/2320ce05aa2d6ea83234469ef86d36481ef962ea/lib/unet.py#L231
    """
    wi: Array
    attn_wo: Array
    norm: Module
    scale: float = static_field()
    num_heads: int = static_field()
    fused_dims: Tuple[int] = static_field()

    def __init__(
        self,
        channels: int,  # input channels effectively the `head_dim`
        *, key: PRNGKey,
        num_heads: int,
        num_groups: Optional[int] = 32,
    ):
        key, ikey, aokey = jax.random.split(key, 3)
        self.scale = math.sqrt(channels) ** -1.0  # scaled dot-product attention factor: 1 / √dₖ
        self.num_heads = num_heads
        self.norm = GroupNorm(num_groups, channels)

        # Fused input projection weights: (Wᵢq, Wᵢᵏ, Wᵢᵛ)
        self.fused_dims = 3 * (num_heads * channels,)
        self.wi = jax.random.normal(ikey, (channels, sum(self.fused_dims)))
        # Output projection weights
        self.attn_wo = jax.random.normal(aokey, (num_heads * channels, channels))

    def __call__(
        self,
        x: Float[Array, "b c e"],
        key: Optional[PRNGKey] = None,
        time: Optional[Array] = None,
    ) -> Float[Array, "b c e"]:
        """
        Notation:
            - `seq_len` is the collapsed spatial dims and
            - `head_dim` is the input `channels`.
        Args:
            time: Unused time arg for sequential processing.
            bias: Attention similarity score bias, e.g. a causal mask.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        # Pre-Norm
        units = self.norm(x)

        # Input Projection
        in_proj = jnp.einsum("... i j, i k -> ... j k", units, self.wi)
        q, k, v = split_fused_proj(in_proj, self.fused_dims)

        # Attention
        # [..., num_heads, seq_len, head_dim]
        q, k, v = map(partial(split_heads, num_heads=self.num_heads), (q, k, v))
        attn = multihead_attn(q, k, v, self.scale)
        concat = merge_heads(attn)  # [..., seq_len, (num_heads * head_dim)]

        # Output projection
        attn_out = concat @ self.attn_wo  # [..., seq_len, head_dim]
        attn_out = attn_out.reshape(x.shape)

        return x + attn_out


class SinusoidalTimeEmbedding(Module):
    dim: int = static_field()
    max_period: int = static_field()

    def __init__(self, dim: int, max_period: int = 10_000):
        """
        Args:
            dim: The embedding dimension.
            max_period: freq = 1 / max_period
        """
        self.dim = dim
        self.max_period = max_period

    def __call__(
        self, time: Float[Array, "b"], *, key: PRNGKey = None
    ) -> Float[Array, "b c"]:
        """
        Args:
            time: Batch of continuous time steps: [0, 1)
        """
        # "Attent is All You Need" Tensor2Tensor pos encoding version:
        # See: https://github.com/tensorflow/tensor2tensor/pull/177
        # Taken from the Magenta folks 🎶 https://github.com/magenta/music-spectrogram-diffusion/blob/ddfeedac58de6fce6872bf8a547df3c1706d0486/music_spectrogram_diffusion/models/diffusion/diffusion_utils.py#L69
        min_timescale = 1.0
        max_timescale = self.max_period
        num_timescales = float(self.dim // 2)
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / (
            num_timescales - 1.0
        )
        inv_timescales = min_timescale * jnp.exp(
            jnp.arange(0, num_timescales) * -log_timescale_increment
        )
        scaled_time = time[:, None] * inv_timescales[None, :] * 100.0
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], -1)
        return signal


def TimeConditionalEmbedding(in_channels: int, key: PRNGKey, time_channels: Optional[int] = None):
    """`time_channels` is an unused arg to satisfy time-based conditioning API"""
    key, tlin1_key, tlin2_key = jax.random.split(key, 3)
    if time_channels is None:
        time_channels = in_channels * 4
    return nn.Sequential([
        SinusoidalTimeEmbedding(dim=in_channels),
        Linear(in_channels, time_channels, key=tlin1_key),
        SiLU(),
        Linear(time_channels, time_channels, key=tlin2_key),
    ])


# UNet


class Upsample(Module):
    scale_factor: int = static_field()
    mode: str = static_field()

    def __init__(
        self,
        scale_factor: int,
        *,
        mode: Optional[str] = "nearest",
    ):
        """
        Args:
            axis: The channel axis in channel-first data format.
        """
        self.scale_factor = scale_factor
        self.mode = mode

    def __call__(
        self, inputs: Float[Array, "b c e"], key: Optional[PRNGKey] = None
    ) -> Array:
        spatial_axes = inputs.shape[2:]
        feature_axes = inputs.shape[:2]
        spatial_scale_shape = np.array(spatial_axes)
        spatial_scale_shape = (self.scale_factor * spatial_scale_shape).tolist()
        shape = (*feature_axes, *spatial_scale_shape)
        return jax.image.resize(inputs, shape, method=self.mode)


class UpsampleBlock1d(Module):
    block: Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *, key: PRNGKey,
    ):
        self.block = nn.Sequential([
            Upsample(scale_factor=2, mode="nearest"),
            Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_bias=False,
                key=key,
            ),
        ])

    def __call__(
        self,
        x: Array,
        *, key: Optional[PRNGKey] = None,
        time: Optional[Array] = None,
    ) -> Array:
        return self.block(x)


class DownsampleBlock(Module):
    block: Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *, key: PRNGKey,
        stride: Optional[Tuple[int, int]] = 2,
    ):
        self.block = Conv1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=stride,
            padding=1,
            key=key,
        )

    def __call__(
        self,
        x: Array,
        *, key: Optional[PRNGKey] = None,
        time: Optional[Array] = None,
    ) -> Array:
        return self.block(x)


class ResidualTimeBlock(Module):
    block1: nn.Sequential
    block2: nn.Sequential
    shortcut: Module
    time_proj: Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        *, key: PRNGKey,
        num_groups: Optional[int] = 32,
        dropout: Optional[float] = 0.0,
    ):
        key, lin_key, *conv_keys = jax.random.split(key, 4)
        self.time_proj = (
            nn.Sequential([
                SiLU(),
                Linear(time_channels, out_channels, key=lin_key),
            ])
            if time_channels is not None
            else nn.Identity()
        )
        self.block1 = nn.Sequential([
            GroupNorm(num_groups, in_channels),
            SiLU(),
            Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                key=conv_keys[0],
            ),
        ])
        self.block2 = nn.Sequential([
            GroupNorm(num_groups, out_channels),
            SiLU(),
            # TODO: Add `dropout` support.
            # nn.Dropout(dropout),
            Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                key=conv_keys[1],
            ),
        ])
        if in_channels != out_channels:
            key, sc_key = jax.random.split(key)
            self.shortcut = Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                key=sc_key,
            )
        else:
            self.shortcut = nn.Identity()

    def __call__(
        self,
        x: Float[Array, "c e"],
        time: Float[Array, "c"],
        *, key: Optional[PRNGKey] = None,
    ) -> Array:
        h = self.block1(x)
        time = self.time_proj(time)
        time = rearrange(time, "... c -> ... c 1")
        h += time
        h = self.block2(h)
        return self.shortcut(x) + h


class DBlock(Module):
    blocks: nn.Sequential
    out_channels: int = static_field()

    def __init__(
        self,
        channels: int,
        time_channels: int,
        channel_mult: Tuple[int],
        attn_resolutions: Tuple[int],
        *, key: PRNGKey,
        num_heads: Optional[int] = 1,
        num_groups: Optional[int] = 32,
        num_res_blocks: Optional[int] = 3,
    ):
        num_resolutions = len(channel_mult)
        out_channels = in_channels = int(channel_mult[0] * channels)
        blocks = []
        for i_level in range(num_resolutions):
            out_channels = channel_mult[i_level] * in_channels
            for i_block in range(num_res_blocks):
                key, drtb_key = jax.random.split(key)
                blocks.append(
                    ResidualTimeBlock(
                        in_channels,
                        out_channels,
                        time_channels,
                        num_groups=num_groups,
                        key=drtb_key,
                    )
                )
                if attn_resolutions[i_level] is True:
                    key, dsab_key = jax.random.split(key)
                    blocks.append(
                        SelfAttentionBlock(
                            out_channels,
                            num_heads=num_heads,
                            num_groups=num_groups,
                            key=dsab_key,
                        )
                    )
                in_channels = out_channels
            if i_level != num_resolutions - 1:
                key, ds_key = jax.random.split(key)
                blocks.append(DownsampleBlock(in_channels, out_channels, key=ds_key))
                in_channels = out_channels
        self.blocks = blocks
        self.out_channels = out_channels

    def __call__(self, x: Array, time: Array) -> Array:
        h = x
        for block in self.blocks:
            h = block(h, time=time)
        return h


class UBlock(Module):
    blocks: nn.Sequential
    out_channels: int = static_field()

    def __init__(
        self,
        channels: int,
        time_channels: int,
        channel_mult: Tuple[int],
        attn_resolutions: Tuple[int],
        *, key: PRNGKey,
        num_res_blocks: Optional[int] = 3,
        num_heads: Optional[int] = 1,
        num_groups: Optional[int] = 32,
    ):
        num_resolutions = len(channel_mult)
        in_channels = out_channels = channels
        blocks = []
        for i_level in reversed(range(num_resolutions)):
            out_channels = in_channels // channel_mult[i_level]
            for i_block in range(num_res_blocks):
                key, urtb_key = jax.random.split(key)
                blocks.append(
                    ResidualTimeBlock(
                        in_channels,
                        out_channels,
                        time_channels,
                        num_groups=num_groups,
                        key=urtb_key,
                    )
                )
                if attn_resolutions[i_level] is True:
                    key, usab_key = jax.random.split(key)
                    blocks.append(
                        SelfAttentionBlock(
                            out_channels,
                            num_heads=num_heads,
                            num_groups=num_groups,
                            key=usab_key,
                        )
                    )
                in_channels = out_channels
            if i_level != 0:
                key, us_key = jax.random.split(key)
                blocks.append(UpsampleBlock1d(out_channels, out_channels, key=us_key))
                in_channels = out_channels
        self.blocks = blocks
        self.out_channels = out_channels

    def __call__(self, x: Array, time: Array) -> Array:
        h = x
        for block in self.blocks:
            h = block(h, time=time)
        return h


class MBlock(Module):
    blocks: Module

    def __init__(
        self,
        channels: int,
        time_channels: int,
        *, key: PRNGKey,
        num_heads: Optional[int] = 1,
        num_groups: Optional[int] = 32,
    ):
        key, rtb1_key, rtb2_key, sab_key = jax.random.split(key, 4)
        self.blocks = [
            ResidualTimeBlock(
                channels, channels, time_channels, num_groups=num_groups, key=rtb1_key
            ),
            SelfAttentionBlock(
                channels,
                num_heads=num_heads,
                num_groups=num_groups,
                key=sab_key,
            ),
            ResidualTimeBlock(
                channels, channels, time_channels, num_groups=num_groups, key=rtb2_key
            ),
        ]

    def __call__(self, x: Array, time: Array) -> Array:
        h = x
        for block in self.blocks:
            h = block(x, time=time)
        return h


class UNet1d(Module):
    cond_embed: TimeConditionalEmbedding
    in_proj: Conv1d
    down_blocks: DBlock
    middle_blocks: MBlock
    up_blocks: UBlock
    out_proj: Conv1d

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        *, key: PRNGKey,
        bit_width: Optional[int] = None,
        num_groups: Optional[int] = 32,
        num_heads: Optional[int] = 1,
        attn_resolutions: Tuple[int] = (False, False, True, True),
        channel_mult: Optional[Tuple[int]] = (1, 2, 4, 8),
    ):
        """
        Args:
            model_channels: Base channel count for the model.
            attn_resolutions: Tuple of `bool`s that indicate whether to use
                attention at each resolution level.
            channel_mult: Tuple of ints that indicate how to scale the model
                channels at each resolution level.
        """
        assert len(attn_resolutions) == len(channel_mult)

        out_channels = in_channels
        if bit_width is not None:
            # Account for the merged input channel and bit-width dimensions.
            in_channels *= bit_width
            in_channels *= 2  # * 2 because of concatenated noise + estimate arrays
            out_channels *= bit_width

        key, in_key = jax.random.split(key)
        self.in_proj = Conv1d(
            in_channels, model_channels, kernel_size=3, padding=1, key=in_key
        )

        key, tckey = jax.random.split(key)
        time_channels = model_channels * 4
        self.cond_embed = TimeConditionalEmbedding(
            model_channels, time_channels=time_channels, key=tckey
        )

        # Initial block input channels
        block_in_channels = model_channels

        key, dbkey = jax.random.split(key)
        self.down_blocks = DBlock(
            channels=block_in_channels,
            channel_mult=channel_mult,
            time_channels=time_channels,
            attn_resolutions=attn_resolutions,
            num_heads=num_heads,
            num_res_blocks=num_res_blocks,
            num_groups=num_groups,
            key=dbkey,
        )

        block_in_channels = self.down_blocks.out_channels
        key, mbkey = jax.random.split(key)
        self.middle_blocks = MBlock(
            channels=block_in_channels,
            time_channels=time_channels,
            num_heads=num_heads,
            num_groups=num_groups,
            key=mbkey,
        )

        key, ubkey = jax.random.split(key)
        self.up_blocks = UBlock(
            channels=block_in_channels,
            channel_mult=channel_mult,
            time_channels=time_channels,
            attn_resolutions=attn_resolutions,
            num_heads=num_heads,
            num_res_blocks=num_res_blocks,
            num_groups=num_groups,
            key=ubkey,
        )

        block_in_channels = self.up_blocks.out_channels
        key, outkey = jax.random.split(key)
        self.out_proj = nn.Sequential([
            GroupNorm(block_in_channels, block_in_channels),
            SiLU(),
            Conv1d(
                block_in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                key=outkey,
            ),
        ])

    def __call__(
        self,
        x: Float[Array, "b c e"],
        time: Float[Array, "b"],
    ) -> Float[Array, "b c e"]:
        time_embed = self.cond_embed(time)
        h = self.in_proj(x)
        h = self.down_blocks(h, time_embed)
        h = self.middle_blocks(h, time_embed)
        h = self.up_blocks(h, time_embed)
        h = self.out_proj(h)
        return h
