from typing import *

import logging

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections as mlc
import optax
import wandb

import configs
from char_diffusion.diffusion import CharDiffusion, bit_encode
from char_diffusion.unet import UNet1d
from char_diffusion.utils import dataloader, decode, enwik8, save, flatten_dict


logger = logging.getLogger(__name__)



def train(config: mlc.ConfigDict):
    device_count = jax.local_device_count()
    logger.info(f"Devices: {jax.devices()}")

    if config.use_wandb:
        wandb.finish()  # Clear out any previous runs.
        wandb.init(
            project=config.project_name,
            entity="jon-tow",
            name=config.name,
            config=flatten_dict(config.to_dict()),
        )

    datasets = enwik8("./tmp/enwik8")
    dataloaders = {
        "train": dataloader(
            datasets["train"],
            seq_len=config.model.seq_len,
            micro_batch_size=config.train.batch_size,
            max_steps=config.train.max_steps,
            device_count=device_count,
        ),
        "valid": dataloader(
            datasets["valid"],
            seq_len=config.model.seq_len,
            micro_batch_size=config.valid.batch_size,
            max_steps=config.train.max_steps,
            device_count=device_count,
        ),
    }
    train_iter = iter(dataloaders["train"])
    valid_iter = iter(dataloaders["valid"])

    key = jax.random.PRNGKey(config.seed)
    net = UNet1d(
        in_channels=1,
        model_channels=config.model.base_channels,
        key=key,
        bit_width=config.model.bit_width,
        num_res_blocks=3,
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
    diffuser = CharDiffusion(
        num_steps=config.model.num_steps,
        use_self_cond=config.model.use_self_cond,
        optim=optim,
    )

    step_load = 0  # state['step']
    for step in range(step_load, config.train.max_steps):
        batch = next(train_iter)
        batch = np.expand_dims(batch, 2)
        key, next_key = jax.random.split(key)
        net, batch_loss, optim_state = diffuser.train_step_pmap(
            net, batch, optim_state, next_key
        )
        # Log training stats.
        if step % config.train.log_every == 0:
            loss = jnp.mean(batch_loss).item()
            wandb.log({"train/loss": loss}, step=step)
        # Evaluate vqvae and log the validation stats.
        if step % config.train.eval_every == 0:
            key, vkey = jax.random.split(key)
            valid_batch = next(valid_iter)
            valid_batch = np.expand_dims(valid_batch, 2)
            valid_batch_loss = diffuser.eval_step_pmap(net, valid_batch, vkey)
            valid_loss = np.mean(valid_batch_loss).item()
            wandb.log({"valid/loss": valid_loss}, step=step)
        # Generate reconstructions and samples.
        if step % config.train.sample_every == 0 and step != 0:
            key, bkey, skey = jax.random.split(key, 3)
            # Log some samples
            num_samples = 6
            batch = jax.random.randint(
                bkey,
                (num_samples, 1, config.model.seq_len),
                0,
                2**config.model.bit_width,
            )
            batch = bit_encode(batch, config.model.bit_width, 1.0).reshape(
                num_samples, config.model.bit_width, config.model.seq_len
            )
            generation = diffuser.generate(
                net,
                batch,
                num_steps=100,
                bit_width=config.model.bit_width,
                key=skey,
                time_delta=0.1,
            )
            generation = generation.squeeze(1).device_buffer.to_py()
            logger.info(f"Generation IDs:\n{generation}")
            logger.info(f"Generations:\n{[decode(g) for g in generation]}")
        if step % config.train.save_every == 0:
            save(net, optim_state, step, config.checkpoint_path)


if __name__ == "__main__":
    config = configs.char_diffusion_enwik8_config(base_dir="./")
    train(config)
