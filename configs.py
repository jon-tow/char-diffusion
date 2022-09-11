import os
import ml_collections as mlc

from pathlib import Path
from typing import *


def mkdir(*paths: List[str]) -> str:
    """Returns a new directory path made of `paths`."""
    new_dir = os.path.join(*paths)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def get_base_config():
    config = mlc.ConfigDict()
    config.seed = 26
    config.name = ""

    # Training configs.
    train = mlc.ConfigDict()
    train.clip_threshold = 1.0  # The clip gradient norm threshold.
    train.batch_size = 64
    train.max_steps = 500_000
    # The below training configs are in units of 1 step (not epochs).
    train.eval_every = 1_000
    train.log_every = 100
    train.sample_every = 1_000
    train.save_every = 50_000
    train.patience = 2_000  # Early-stopping patience.
    config.train = train

    # Validation.
    valid = mlc.ConfigDict()
    valid.batch_size = 128
    config.valid = valid

    # Attach configs not declared here outside.
    return config


def optimizer_config():
    # ADAM optimization configs.
    config = mlc.ConfigDict()
    config.lr = 1e-4
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.99
    return config


def char_diffusion_enwik8_config(base_dir: str):
    config = get_base_config()

    # Analog bit encoding settings
    config.model = mlc.ConfigDict()
    config.model.name = "char-diffusion"
    config.model.seq_len = 64
    config.model.bit_width = 8
    config.model.scale = 1.0
    config.model.num_steps = 1_000
    config.model.use_self_cond = True  # False
    config.model.ema_decay = 0.9999
    config.model.num_res_blocks = 3
    config.model.base_channels = 128

    # Add optimizier config
    config.optim = optimizer_config()

    # Add dataset config
    config.dataset = mlc.ConfigDict()
    config.dataset.name = "enwik8"
    config.dataset.dir = mkdir(base_dir, "data")

    # Add model config
    # Add config name and working directory checkpoints, logs, etc.
    config.name = f"{config.model.name}_{config.dataset.name}"
    config.sample_dir = mkdir(Path(base_dir), "samples", config.name)

    checkpoint_dir = mkdir(
        Path(base_dir), "checkpoints", "baseline-char-diffusion-enwik8"
    )
    config.checkpoint_path = str(Path(checkpoint_dir) / f"{config.name}.eqx")
    Path(config.checkpoint_path).touch(exist_ok=True)

    # Logging
    config.use_wandb = True
    config.project_name = f"{config.name}"
    return config
