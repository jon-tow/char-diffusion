import os
import ml_collections as mlc

from pathlib import Path
from typing import *


def get_base_config() -> mlc.ConfigDict:
    config = mlc.ConfigDict()
    config.seed = 26
    config.name = ""

    # Training configs
    train = mlc.ConfigDict()
    train.batch_size = 32
    train.max_steps = 200_000
    train.clip_threshold = 1.0  # The clip gradient norm threshold.
    train.resume = False  # Whether to resume training from a checkpoint.
    # NOTE: The below training configs are in units of steps (not epochs).
    train.eval_every = 1_000
    train.log_every = 100
    train.sample_every = 1_000
    train.save_every = 10_000
    train.patience = 2_000  # Early-stopping patience.
    config.train = train

    # Validation
    valid = mlc.ConfigDict()
    valid.batch_size = 32
    config.valid = valid

    # Attach configs not declared here outside
    return config


def optimizer_config() -> mlc.ConfigDict:
    # ADAM optimization configs
    config = mlc.ConfigDict()
    config.lr = 2e-4
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.99
    return config


def char_diffusion_base_config(
    dataset_path: Optional[str] = None, 
    id: Optional[int] = 0, 
) -> mlc.ConfigDict:
    config = get_base_config()

    # Add model config
    config.model = mlc.ConfigDict()
    config.model.name = "char-diffusion"
    config.model.seq_len = 256
    config.model.max_gen_len = 128
    config.model.base_channels = 296
    # Bit Diffusion settings
    config.model.bit_width = 8
    config.model.use_self_cond = True
    config.model.ema_decay = 0.9999
    config.model.scale = 1.0
    config.model.num_res_blocks = 3
    config.model.num_steps = 2_000
    config.model.num_gen_steps = 1_000

    # Add optimizier config
    config.optim = optimizer_config()

    # Add dataset config
    config.dataset = mlc.ConfigDict()
    config.dataset.name = Path(dataset_path).stem if dataset_path else "none"
    config.dataset.path = dataset_path

    # Add config name and working directory checkpoints, logs, etc.
    config.name = f"{config.model.name}_{config.dataset.name}"
    config.output_dir = f"./checkpoints/{config.name}-{id}"
    config.checkpoint_path = None

    # Logging
    config.use_wandb = True
    config.wandb_id = None
    config.wandb_entity = ""
    config.wandb_project_name = f"{config.name}"

    return config