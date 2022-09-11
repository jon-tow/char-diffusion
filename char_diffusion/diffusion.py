from typing import *
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp
import numpy as np
import optax

from equinox import Module
from functools import partial
from einops import rearrange

from .custom_layers import left_broadcast_to
from .losses import mse


PRNGKey = NewType("PRNGKey", jax._src.prng.PRNGKeyArray)


# Bit ops


def int2bit(inputs: Array, width: int = 8) -> Array:
    """Converts array of integers into corresponding binary bits."""
    inputs = rearrange(inputs, "... w -> ... w 1")
    # If you don't flip, the lsb is the msb.
    bitstring = jnp.flip(jnp.arange(width), -1)
    outputs = jnp.right_shift(inputs, bitstring)
    outputs = jnp.fmod(outputs, 2)
    return outputs


def bit2int(inputs: Array, width: int = 8) -> Array:
    """Convert binary bits into the corresponding integers."""
    int_inputs = inputs.astype(jnp.int32)
    # If you don't flip, the lsb is the msb.
    bitstring = jnp.flip(jnp.arange(width), -1)
    outputs = jnp.sum(int_inputs * (2**bitstring), -1)
    return outputs


def bit_encode(
    inputs: Array,
    width: int,
    scale: float,
) -> Array:
    discrete_bits = int2bit(inputs, width)
    analog_bits = discrete_bits.astype(jnp.float32)
    # Shift-and-scale, e.g.: {0, 1} -> {-1, 1}
    analog_bits = (analog_bits * 2 - 1) * scale
    analog_bits = rearrange(analog_bits, "c ... i -> (c i) ...")
    return analog_bits


bit_encode = jax.vmap(bit_encode, in_axes=(0, None, None))


def bit_decode(bits: Array, width: int) -> Array:
    """Threshold and convert to integers from analog bits"""
    bits = rearrange(bits, "(c i) ...  -> c ... i", i=width)
    return bit2int(bits > 0.0, width)


bit_decode = jax.vmap(bit_decode, in_axes=(0, None))


# Diffusion scheduler


def cosine_alpha_bar(
    time: float,
    offset: Optional[float] = 0.0002,
) -> Array:
    """Cosine noise-variance ᾱ scheduler (ᾱ[t] = Πᵗα[i] where α[i] = (1 - β[i]))
    for continuous time parameterization.

    Reference: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models".
        2021. https://arxiv.org/pdf/2102.09672.pdf

    Args:
        offset: Small offset to prevent βₜ from beeing too small near
            t = 0.
    """
    return jnp.cos(((time + offset) / (1 + offset)) * (jnp.pi / 2)) ** 2


def cosine_alpha_bar_schedule(
    offset: Optional[float] = 0.0002,
) -> Array:
    def scheduler(num_steps: float):
        return cosine_alpha_bar(time=num_steps, offset=offset)

    return scheduler


def sqrt_alpha_bar(
    time: float,
    # max_time: int,  # Use if time is NOT in [0, 1)
    offset: Optional[float] = 1e-4,
):
    """Square-root noise schedule - useful for language modeling.

    Reference: Li et al. "Diffusion-LM Improves Controllable Text Generation"
        2022. https://arxiv.org/pdf/2205.14217.pdf#page=15

    Args:
        time: Continous time-step in [0, 1)
        offset: Start noise level.
    """
    # Normalize if your input is not in [0, 1)
    # return 1.0 - jnp.sqrt((time / max_time) + offset)
    return 1.0 - jnp.sqrt(time + offset)


def sqrt_alpha_bar_schedule(
    offset: Optional[float] = 1e-4,
) -> Array:
    def scheduler(num_steps: float):
        return sqrt_alpha_bar(time=num_steps, offset=offset)

    return scheduler


# Diffusion


class CharDiffusion:
    """
    Reference: Chen et al. "Analog Bits: Generating Discrete Data using
        Diffusion Models with Self-Conditioning". 2022.
        https://arxiv.org/pdf/2208.04202.pdf
    """

    def __init__(
        self,
        num_steps: int,
        bit_width: Optional[int] = 8,
        bit_scale: Optional[float] = 1.0,
        use_self_cond: Optional[bool] = False,
        gamma_schedule: Optional[Callable] = cosine_alpha_bar_schedule(),
        optim: Optional[optax.GradientTransformation] = None,
        channel_axis: Optional[int] = 1,
    ):
        self.channel_axis = channel_axis
        self.num_steps = num_steps
        self.bit_width = bit_width
        self.bit_scale = bit_scale
        self.use_self_cond = use_self_cond
        self.optim = optim

        # Continuous time parameterization
        self.gamma = gamma_schedule  # γ = ᾱ

        def train_step(
            net: Module,
            x: Float[Array, "b c ..."],
            optim_state: Tuple,
            key: PRNGKey,
        ) -> Tuple[Array, float, Array]:
            """Batched train-step"""
            loss, grad = jax.value_and_grad(
                lambda n, x, k: jnp.mean(self.loss_fn(n, x, k)), allow_int=True
            )(net, x, key)
            grad = jax.lax.pmean(grad, axis_name="batch")
            updates, optim_state = self.optim.update(grad, optim_state, net)
            net = optax.apply_updates(net, updates)
            return net, loss, optim_state

        self.train_step = train_step
        self.train_step_pmap = jax.pmap(
            self.train_step,
            in_axes=(None, 0, None, None),
            out_axes=(None, 0, None),
            axis_name="batch",
        )

        def eval_step(net: Module, x: Array, key: PRNGKey) -> Array:
            loss = self.loss_fn(net, x, key)
            return jnp.asarray(loss)

        self.eval_step = eval_step
        self.eval_step_pmap = jax.pmap(
            self.eval_step,
            in_axes=(None, 0, None),
        )

    def loss_fn(self, net: Module, x: Float[Array, "b c ..."], key: PRNGKey) -> Array:
        key, tkey = jax.random.split(key)
        batch_size = x.shape[0]

        # Binary encoding: discrete data to analog bits
        bits = bit_encode(x, self.bit_width, self.bit_scale)

        # Select random timestep
        time = jax.random.uniform(tkey, (batch_size,))
        noisy_bits = self.corrupt(bits, time, key=key)

        # Compute self-conditioning estimate
        cond_bits = jnp.zeros_like(noisy_bits, dtype=noisy_bits.dtype)
        rand_self_cond = np.random.rand()
        cond_bits = jax.lax.cond(
            self.use_self_cond and (rand_self_cond > 0.5),
            partial(self.self_cond_estimate, net=net, time=time),
            lambda noisy_bits, cond_bits: cond_bits,
            noisy_bits,
            cond_bits,
        )

        # Predict and compute loss
        pred_bits = net(
            jnp.concatenate([noisy_bits, cond_bits], self.channel_axis), time
        )
        loss = mse(pred_bits, targets=bits)
        return loss

    def corrupt(self, x: Array, time: int, key: PRNGKey) -> Array:
        """q sampler: q(xₜ | xₒ) ~ N(xₒ * Π(√(1-β)), 1 - Π(1 - β))
        Arbitrary time sampler for forward diffusion processing (corruption).
        Reference: Ho et al. 2020
        """
        key, nkey = jax.random.split(key)
        noise = jax.random.normal(nkey, x.shape)  # ϵ

        signal_rate = jnp.sqrt(self.gamma(time))[:, None]
        noise_rate = jnp.sqrt(1 - self.gamma(time))

        signal_rate = left_broadcast_to(signal_rate, x.shape)
        noise_rate = left_broadcast_to(noise_rate, x.shape)
        return signal_rate * x + noise_rate * noise

    def self_cond_estimate(
        self, noisy_bits: Array, pred_bits: Array, net: Module, time: Array
    ) -> Array:
        cond_bits = net(
            jnp.concatenate([noisy_bits, pred_bits], self.channel_axis), time
        )
        cond_bits = jax.lax.stop_gradient(cond_bits)
        return cond_bits

    def generate(
        self,
        net: Module,
        x: Float[Array, "b c e"],
        num_steps: int,
        bit_width: int,
        key: PRNGKey,
        time_delta: int = 0,
    ) -> Array:
        """p sampler
        Sampler for the reverse diffusion process (denoising), i.e. predicts
        the initial x, xₒ.

        Args:
            x: Junk array that we can gather output shapes from.
            time_delta: Asymmetric time interval shift, t → (t - Δ)
        """
        key, tkey = jax.random.split(key, 2)

        x_t = jax.random.normal(tkey, x.shape)
        pred = jnp.zeros_like(x_t)

        def _generate_body(
            step: int, state: Tuple[Array, Array]
        ) -> Tuple[Array, Array]:
            pred, x_t_prev = state
            # Get time for current and next states
            time_now = jnp.array([1 - step / num_steps])
            time_next = jnp.array(
                [jnp.maximum(1 - (step + 1 + time_delta) / num_steps, 0.0)]
            )
            # Predict x_0
            pred = jax.lax.cond(
                self.use_self_cond,
                lambda xtp: net(
                    jnp.concatenate([xtp, pred], self.channel_axis), time_now
                ),
                lambda xtp: net(
                    jnp.concatenate([xtp, jnp.zeros_like(xtp)], self.channel_axis),
                    time_now,
                ),
                x_t_prev,
            )
            # Estimate x at time_next
            x_t_next = self.ddim_step(x_t_prev, pred, time_now, time_next)
            return pred, x_t_next

        init_state = (pred, x_t)
        pred, x_t = jax.lax.fori_loop(0, num_steps, _generate_body, init_state)

        # Binary decoding: analog bits to discrete data - bit2int(pred > 0.0)
        x_int = bit_decode(pred, bit_width)
        return x_int

    def ddim_step(self, x_t, x_pred, time_now, time_next) -> Array:
        """Denoising diffusion implicit model step with η = 0
        Estimates x at time_next with the DDIM updating rule
        References:
        - Song et al. "Denoising Diffusion Implicit Models" 2020.
          https://arxiv.org/pdf/2010.02502.pdf
        - Lilian Weng. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#speed-up-diffusion-model-sampling
        """
        gamma_now = self.gamma(time_now)
        gamma_next = self.gamma(time_next)
        noise = jax.lax.rsqrt(1 - gamma_now) * (x_t - jnp.sqrt(gamma_now) * x_pred)
        x_next = jnp.sqrt(gamma_next) * x_pred + jnp.sqrt(1 - gamma_next) * noise
        return x_next
