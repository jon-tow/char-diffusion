"""Custom Batched Layers: `equinox` does not support batched layers by default."""
from typing import *
from jaxtyping import Float, Array

import math
import jax
import jax.numpy as jnp

from equinox import Module, static_field


PRNGKey = NewType("PRNGKey", jax._src.prng.PRNGKeyArray)


def _ntuple(n: int) -> Callable:
    def parse(x: Any) -> tuple:
        if isinstance(x, Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(
                    f"Length of {x} (length = {len(x)}) is not equal to {n}"
                )
        else:
            import itertools

            return tuple(itertools.repeat(x, n))

    return parse


def left_broadcast_to(arr: Array, shape: Tuple[int]):
    arr = arr.reshape(arr.shape + (1,) * (len(shape) - arr.ndim))
    return jnp.broadcast_to(arr, shape)


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool,
) -> jax.lax.ConvDimensionNumbers:
    """Create a `lax.ConvDimensionNumbers` for the given inputs.

    Reference:
    - DeepMind `dm-haiku`
    """
    num_dims = num_spatial_dims + 2

    if channels_last:
        spatial_dims = tuple(range(1, num_dims - 1))
        image_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        image_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return jax.lax.ConvDimensionNumbers(
        lhs_spec=image_dn, rhs_spec=kernel_dn, out_spec=image_dn
    )


class Conv(Module):
    """Batched Convolution Layer
    
    Reference:
    - https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/conv.py
    """
    num_spatial_dims: int = static_field()
    weight: Array
    bias: Optional[Array]
    in_channels: int = static_field()
    out_channels: int = static_field()
    kernel_size: Tuple[int, ...] = static_field()
    stride: Tuple[int, ...] = static_field()
    padding: Tuple[Tuple[int, int], ...] = static_field()
    dilation: Tuple[int, ...] = static_field()
    groups: int = static_field()
    use_bias: bool = static_field()
    dimension_numbers: jax.lax.ConvDimensionNumbers = static_field()

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *, key: PRNGKey,
        **kwargs,
    ):
        super().__init__(**kwargs)
        wkey, bkey = jax.random.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        dilation = parse(dilation)

        if in_channels % groups != 0:
            raise ValueError(
                f"`in_channels` (={in_channels}) must be divisible "
                f"by `groups` (={groups})."
            )

        grouped_in_channels = in_channels // groups
        lim = 1 / jnp.sqrt(grouped_in_channels * jnp.prod(jnp.array(kernel_size)))
        # 'OIC'
        weight_shape = (out_channels, grouped_in_channels) + kernel_size
        self.weight = jax.random.uniform(
            wkey,
            weight_shape,
            minval=-lim,
            maxval=lim,
        )
        if use_bias:
            self.bias = jax.random.uniform(
                bkey,
                shape=(out_channels,) + (1,) * (num_spatial_dims),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{num_spatial_dims}."
            )
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.dimension_numbers = to_dimension_numbers(
            num_spatial_dims, channels_last=False, transpose=False
        )

    def __call__(
        self, inputs: Array, *, key: Optional["jax.random.PRNGKey"] = None
    ) -> Array:
        """
        Returns:
            An array of shape (batch, out_channels, new_dim_1, ..., new_dim_N)`.
        """
        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvND needs to have rank in {allowed_ranks},"
                f" but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)

        if self.in_channels % self.groups != 0:
            raise ValueError(
                f"Inputs channels {inputs.shape[1]} "
                f"should be a multiple of feature_group_count "
                f"{self.groups}"
            )

        out = jax.lax.conv_general_dilated(
            lhs=inputs,
            rhs=self.weight,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        if self.use_bias:
            out = out + self.bias

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class Conv1d(Conv):
    """('NCW', 'OIW', 'NCW')"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        use_bias=True,
        *, key,
        **kwargs,
    ):
        super().__init__(
            num_spatial_dims=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )


class Linear(Module):
    weight: Array
    bias: Optional[Array]
    in_features: int = static_field()
    out_features: int = static_field()
    use_bias: bool = static_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *, key: PRNGKey,
    ):
        super().__init__()
        wkey, bkey = jax.random.split(key, 2)
        lim = 1 / math.sqrt(in_features)
        self.weight = jax.random.uniform(
            wkey, (in_features, out_features), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jax.random.uniform(
                bkey, (out_features,), minval=-lim, maxval=lim
            )
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self, inputs: Float[Array, "b i"], *, key: PRNGKey = None
    ) -> Float[Array, "b o"]:
        inputs = jnp.dot(inputs, self.weight)
        if self.bias is not None:
            inputs = inputs + self.bias
        return inputs


class GroupNorm(Module):
    num_groups: int = static_field()
    num_channels: int = static_field()
    eps: float = static_field()
    affine: bool = static_field()
    channel_index: bool = static_field()
    weight: Array
    bias: Array

    def __init__(
        self,
        num_groups: int,
        num_channels: Optional[int] = None,
        eps: Optional[float] = 1e-5,
        affine: Optional[bool] = True,
        channel_index: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (num_channels is not None) and (num_channels % num_groups != 0):
            raise ValueError(
                "The number of num_groups must divide the number of num_channels."
            )
        if (num_channels is None) and affine:
            raise ValueError(
                "The number of num_channels should be specified if `affine=True`"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.channel_index = channel_index
        self.weight = jnp.ones((1, num_channels)) if affine else None
        self.bias = jnp.zeros((1, num_channels)) if affine else None

    def __call__(
        self,
        inputs: Float[Array, "b c ..."],
        *, key: Optional["jax.random.PRNGKey"] = None,
    ) -> Array:
        num_channels = inputs.shape[self.channel_index]
        y = inputs.reshape(
            inputs.shape[0],
            self.num_groups,
            num_channels // self.num_groups,
            *inputs.shape[2:],
        )
        axis = [i for i in range(2, len(y.shape))]
        mean = jnp.mean(y, axis=axis, keepdims=True)
        variance = jnp.var(y, axis=axis, keepdims=True)
        out = (y - mean) * jax.lax.rsqrt(variance + self.eps)
        out = out.reshape(inputs.shape)
        if self.affine:
            weight = left_broadcast_to(self.weight, out.shape)
            bias = left_broadcast_to(self.bias, out.shape)
            out = weight * out + bias
        return out


class LayerNorm(Module):
    gamma: jnp.ndarray
    eps: float = static_field()

    def __init__(self, dim, eps = 1e-5):
        self.gamma = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean_of_squares = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        variance = mean_of_squares - jnp.square(mean)
        inv = jax.lax.rsqrt(variance + self.eps)
        return inv * (x - mean) * self.gamma