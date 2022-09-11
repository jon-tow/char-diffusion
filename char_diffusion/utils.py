from jaxtyping import PyTree, Array
from typing import *

import numpy as np


def flatten_dict(d: dict, parent_key: str = "") -> dict:
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


def save(model: PyTree, optim_state: PyTree, step: int, path: str):
    """Saves an `equinox` model to the specified file path."""
    import equinox as eqx

    eqx.tree_serialise_leaves(path, (model, optim_state, step))


def load(model: PyTree, path: str) -> Tuple[PyTree, PyTree, int]:
    import equinox as eqx

    return eqx.tree_deserialise_leaves(path, model)


def enwik8(
    path: str,
    num_train: int = int(90e6),
    num_valid: int = int(5e6),
    num_test: int = int(5e6),
) -> Mapping[str, Array]:
    # wget http://mattmahoney.net/dc/enwik8.zip -P ./tmp
    # unzip ./tmp/enwik8.zip -d ./tmp
    with open(path, mode="rb") as f:
        text = f.read(num_train + num_valid + num_test)
        data = np.frombuffer(text, dtype=np.uint8)
    train, valid, test = np.split(data, [num_train, num_train + num_valid])
    return dict(train=train, valid=valid, test=test)


def dataloader(
    dataset: Array,
    seq_len: int,
    micro_batch_size: int,
    device_count: Optional[int] = 1,
    max_steps: int = 5e6,
    rng: np.random.Generator = np.random.default_rng(2694),
) -> Array:
    """Returns a random batch of data from the specified dataset.
    From @lucidrains
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
