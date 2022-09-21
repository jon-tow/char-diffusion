from pathlib import Path
from jaxtyping import PyTree, Array
from typing import *

import logging
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import equinox.experimental as experimental


# PyTree utils


def count(tree: PyTree) -> int:
    return sum(t.size for t in jax.tree_util.tree_leaves(tree))


def flatten_dict(d: dict, parent_key: Optional[str] = "") -> dict:
    """
    Flattens a dict-of-dicts, replacing any nested key names with that name
    prepended with the parents' key names.
    """
    flat_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_d.update(flatten_dict(v, parent_key=f"{k}_"))
        else:
            flat_d[f"{parent_key}{k}"] = v
    return flat_d


def default_deserialise_filter_spec(
    f: str, x: Any, allow_pickle: Optional[bool] = True
) -> Any:
    """Override default deserialise filter spec to allow loading pickled arrays."""
    if isinstance(x, jnp.ndarray):
        return jnp.load(f, allow_pickle=allow_pickle)
    elif isinstance(x, np.ndarray):
        return np.load(f, allow_pickle=allow_pickle)
    elif isinstance(x, (bool, float, complex, int)):
        return np.load(f, allow_pickle=allow_pickle).item()
    elif isinstance(x, experimental.StateIndex):
        # Make a new StateIndex. If we happen to load some state then we don't
        # want to affect the `like` as a side-effect.
        y = experimental.StateIndex(inference=x.inference)
        saved_value = np.load(f, allow_pickle=allow_pickle).item()
        assert isinstance(saved_value, bool)
        if saved_value:
            is_array = np.load(f, allow_pickle=allow_pickle).item()
            assert isinstance(is_array, bool)
            if is_array:
                value = jnp.load(f, allow_pickle=allow_pickle)
            else:
                tuple_length = np.load(f, allow_pickle=allow_pickle).item()
                assert isinstance(tuple_length, int)
                value = tuple(jnp.load(f, allow_pickle=allow_pickle) for _ in range(tuple_length))
            experimental.set_state(y, value)
        return y
    else:
        return x


def load_state_dict(
    path: str,
    tree: PyTree,
    filter_spec: Optional[Callable] = default_deserialise_filter_spec,
) -> Tuple[PyTree, PyTree, int]:
    """Load a PyTree from the specified file path."""
    return eqx.tree_deserialise_leaves(path, tree, filter_spec=filter_spec)


def save(path: str, tree: PyTree):
    """Saves a PyTree to the specified file path."""
    if not os.path.exists(path):
        os.makedirs(Path(path).parent, exist_ok=True)
        Path(path).touch()
    eqx.tree_serialise_leaves(path, tree)


# Dataset utils


def mahoney_dataset(
    path: str,
    num_train: Optional[int] = int(90e6),
    num_valid: Optional[int] = int(5e6),
    num_test: Optional[int] = int(5e6),
) -> Mapping[str, Array]:
    """Splits a Matth Mahoney dataset, e.g. text or enwik8.
    ```
        wget http://mattmahoney.net/dc/text8.zip -P ./tmp
        unzip ./tmp/text8.zip -d ./tmp   
    ```
    """
    with open(path, mode="rb") as f:
        text = f.read(num_train + num_valid + num_test)
        data = np.frombuffer(text, dtype=np.uint8)
    train, valid, test = np.split(data, [num_train, num_train + num_valid])
    return dict(train=train, valid=valid, test=test)


def text_dataset(
    path: str,
    num_train: Optional[float] = 0.9,
    num_valid: Optional[float] = 0.06,
) -> Mapping[str, Array]:
    """Splits a `.txt` dataset that can be read in-memory.
    Args:
        path: Path to a `.txt` file that can fit in-memory.
    """
    with open(path, mode="r") as f:
        text = f.read()
        text = " ".join(text.splitlines())
        text = text.replace("   ", " ")
        text = text.replace("  ", " ")
        text = text.strip()
        data = np.fromstring(text, dtype=np.uint8)
    train, valid, test = np.split(data, [
        int(num_train * len(data)),
        int((num_train + num_valid) * len(data)),
    ])
    return dict(train=train, valid=valid, test=test)


def dataloader(
    dataset: Array,
    seq_len: int,
    micro_batch_size: int,
    device_count: Optional[int] = 1,
    max_steps: Optional[int] = 5e6,
    rng: Optional[np.random.Generator] = np.random.default_rng(9426),
) -> Array:
    """Returns a shuffled dataset iterator from the specified dataset.
    Reference: @lucidrains
    """
    i = 0
    while i < max_steps:
        total_seq_len = dataset.shape[0]
        batch_size = micro_batch_size * device_count
        base_arange = np.arange(seq_len)
        start_indices = rng.integers(
            low=0, high=total_seq_len - seq_len, size=batch_size
        )
        token_indices = start_indices[:, None] + base_arange
        tokens = dataset[token_indices].reshape(device_count, micro_batch_size, -1)
        yield tokens


def decode(tokens: List[int]) -> str:
    return "".join(chr(max(t, 32)) for t in tokens)


# File utils


def mkdir(*paths: List[str]) -> str:
    """Returns a new directory path made of `paths`."""
    new_dir = os.path.join(*paths)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def init_logger(
    logger: logging.Logger, output_dir: str, stdout_only=False
):
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if not stdout_only:
        file_handler = logging.FileHandler(
            filename=os.path.join(output_dir, 'run.log'))
        handlers.append(file_handler)
    logger.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(handler)
