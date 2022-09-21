import jax
import ml_collections as mlc
import optax

import char_diffusion as cd
from char_diffusion import utils
from char_diffusion import configs
from char_diffusion.diffusion import get_schedule


def generate(config: mlc.ConfigDict):
    assert config.checkpoint_path is not None, \
        "Must provide a checkpoint path to generate samples."

    key = jax.random.PRNGKey(config.seed)
    net = cd.UNet1d(
        in_channels=1,
        model_channels=config.model.base_channels,
        key=key,
        bit_width=config.model.bit_width,
        num_res_blocks=config.model.num_res_blocks,
        num_heads=config.model.num_heads,
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

    net, optim_state, step_state = utils.load_state_dict(
        path=config.checkpoint_path,
        tree=(net, optim_state, step_state)
    )

    diffuser = cd.CharDiffusion(
        num_steps=config.model.num_steps,
        use_self_cond=config.model.use_self_cond,
        gamma_schedule=get_schedule(config.model.schedule),
        optim=optim,
    )

    key, gen_key = jax.random.split(key)
    num_samples = 8
    generation = diffuser.generate(
        net,
        shape=(num_samples, config.model.bit_width, config.model.seq_len),
        num_steps=config.model.num_gen_steps,
        bit_width=config.model.bit_width,
        key=gen_key,
        time_delta=config.model.time_delta,
    )
    generation = generation.squeeze(1).device_buffer.to_py()
    print(f"Generation IDs:\n{generation}")
    print(f"Generations:\n{[cd.utils.decode(g) for g in generation]}")

if __name__ == "__main__":
    config = configs.char_diffusion_base_config()
    config.seed = 9999
    config.model.num_gen_steps = 2_000
    config.model.schedule = "cosine"
    config.checkpoint_path = ""
    generate(config)