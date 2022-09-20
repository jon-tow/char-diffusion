from typing import *

import os
import sys
import logging

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections as mlc
import optax
import wandb

import char_diffusion as cd
import char_diffusion.configs as configs
from char_diffusion.diffusion import get_schedule
from char_diffusion.utils import *


logger = logging.getLogger(__name__)


def train(config: mlc.ConfigDict):
    device_count = jax.local_device_count()
    logger.info(f"Devices: {jax.devices()}")

    if config.use_wandb:
        wandb.finish()  # Clear out any previous runs.
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            name=config.name,
            config=flatten_dict(config.to_dict()),
            id=config.wandb_id,
        )

    if config.dataset.name in ['enwik8', 'text8']:
        datasets = mahoney_dataset(config.dataset.path)
    else:
        datasets = text_dataset(config.dataset.path)
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

    # TODO: Update CharDiffusion so we don't have to specify `bit_width` twice;
    # (once in the unet and once in the diffuser).
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
    optim = optax.chain(
        optax.clip_by_global_norm(
            config.optim.clip_threshold,
        ),
        optax.adam(
            config.optim.lr,
            b1=config.optim.adam_beta1,
            b2=config.optim.adam_beta2,
            eps=1e-8,
        )
    )
    optim_state = optim.init(net)
    step_state = 0
    if (
        config.train.resume 
        and config.checkpoint_path is not None
        and Path(config.checkpoint_path).exists()
    ):
        net, optim_state, step_state = load_state_dict(
            path=config.checkpoint_path,
            tree=(net, optim_state, step_state)
        )
    elif config.train.resume and Path(config.output_dir).exists():
        net, optim_state, step_state = load_state_dict(
            path=os.path.join(config.output_dir, "checkpoint", "latest", "checkpoint.eqx"),
            tree=(net, optim_state, step_state)
        )

    logger.info(f"Network parameter count: ~ {format(count(net), ',')}")
    logger.info(f"Starting Step: {step_state}")
    logger.info(f"Config:\n{config}")
    
    diffuser = cd.CharDiffusion(
        num_steps=config.model.num_steps,
        bit_width=config.model.bit_width,
        use_self_cond=config.model.use_self_cond,
        gamma_schedule=get_schedule(config.model.schedule),
        optim=optim,
    )

    for step in range(step_state, config.train.max_steps):
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
            info = f"Step: {step}/{config.train.max_steps} | Loss: {loss:.5f}"
            logger.info(info)
        # Evaluate vqvae and log the validation stats.
        if step % config.train.eval_every == 0:
            key, valid_key = jax.random.split(key)
            valid_batch = next(valid_iter)
            valid_batch = np.expand_dims(valid_batch, 2)
            valid_batch_loss = diffuser.eval_step_pmap(net, valid_batch, valid_key)
            valid_loss = np.mean(valid_batch_loss).item()
            wandb.log({"valid/loss": valid_loss}, step=step)
            save(
                path=os.path.join(config.output_dir, "checkpoint", "latest", "checkpoint.eqx"),
                tree=(net, optim_state, step)
            )
        # Generate reconstructions and samples.
        if step % config.train.sample_every == 0 and step != 0:
            key, gen_key = jax.random.split(key)
            num_samples = 8
            samples = diffuser.generate(
                net,
                shape=(num_samples, config.model.bit_width, config.model.max_gen_len),
                num_steps=config.model.num_gen_steps,
                bit_width=config.model.bit_width,
                key=gen_key,
                time_delta=config.model.time_delta,
            )
            samples = samples.squeeze(1).device_buffer.to_py()
            sample_log = "\nSamples:\n"
            for sample in samples:
                sample_log += f"➜ {decode(sample)}\n"
            logger.info(sample_log)
        if step % config.train.save_every == 0 and step != 0:
            save(
                path=os.path.join(config.output_dir, "checkpoint", f"step-{step}", "checkpoint.eqx"),
                tree=(net, optim_state, step)
            )


if __name__ == "__main__":
    config = configs.char_diffusion_base_config(
        dataset_path="./tmp/war_and_peace.txt", #"./tmp/linux.txt",
        id=np.random.randint(0, 1e5),
    )
    config.wandb_entity = "jon-tow"
    # config.wandb_id = "2itvppsj"
    # config.train.resume = True
    # config.output_dir = "/fsx/guac/char-diffusion/checkpoints/char-diffusion_war_and_peace-35535"
    # config.checkpoint_path = "/fsx/guac/char-diffusion/checkpoints/char-diffusion_war_and_peace-35535/checkpoint/latest/"

    os.makedirs(config.output_dir, exist_ok=True)

    init_logger(logger, config.output_dir)
    train(config)
