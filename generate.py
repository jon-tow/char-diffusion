import os

import jax
import jax.numpy as jnp
import equinox as eqx
import ml_collections as mlc
import numpy as np
import optax

import char_diffusion as cd
import configs

from char_diffusion import utils


def main(config: mlc.ConfigDict):
    key = jax.random.PRNGKey(config.seed)
    net = cd.UNet1d(
        in_channels=1,
        model_channels=config.model.base_channels,
        key=key,
        bit_width=config.model.bit_width,
        num_res_blocks=3,
        num_heads=1,
        num_groups=4,
        attn_resolutions=(False, False, True),
        channel_mult=(1, 2, 4),
    )
    optim = optax.adam(
        config.optim.lr,
        b1=config.optim.adam_beta1,
        b2=config.optim.adam_beta2,
        eps=1e-8,
    )
    optim_state = optim.init(net)
    step_state = 0
    if config.checkpoint_path:
        net, optim_state, step_state = utils.load_state_dict(
            config.checkpoint_path,
            (net, optim_state, step_state)
        )

    diffuser = cd.CharDiffusion(
        num_steps=config.model.num_steps,
        use_self_cond=config.model.use_self_cond,
        optim=optim,
    )

    key, gen_key = jax.random.split(key)
    num_samples = 8
    generation = diffuser.generate(
        net,
        shape=(num_samples, config.model.bit_width, config.model.seq_len),
        num_steps=config.model.num_generation_steps,
        bit_width=config.model.bit_width,
        key=gen_key,
        time_delta=0,
    )
    generation = generation.squeeze(1).device_buffer.to_py()
    print(f"Generation IDs:\n{generation}")
    print(f"Generations:\n{[cd.utils.decode(g) for g in generation]}")

if __name__ == "__main__":
    config = configs.char_diffusion_text8_config(base_dir="./")
    config.model.bit_width = 8
    config.checkpoint_path = "/fsx/guac/char-diffusion/checkpoints/baseline-char-diffusion-text8/char-diffusion_shakespeare.eqx"
    main(config)