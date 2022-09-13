from jaxtyping import PyTree, Array
from typing import *

import equinox as eqx
import equinox.experimental as experimental
import jax.numpy as jnp
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


def save(path: str, tree: PyTree):
    """Saves an `equinox` model to the specified file path."""
    eqx.tree_serialise_leaves(path, tree)


def load_state_dict(path: str, tree: PyTree) -> Tuple[PyTree, PyTree, int]:
    return eqx.tree_deserialise_leaves(path, tree, filter_spec=default_deserialise_filter_spec)


def default_deserialise_filter_spec(
    f, x: Any, allow_pickle: bool = True
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


def mahoney_dataset(
    path: str,
    num_train: int = int(90e6),
    num_valid: int = int(5e6),
    num_test: int = int(5e6),
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
    num_train: float = 0.9,
    num_valid: int = 0.06,
) -> Mapping[str, Array]:
    """Splits a `.txt` dataset that can be read in-memory.
    Args:
        path: Path to a `.txt` file that can fit in-memory.
    """
    with open(path, mode="r") as f:
        text = f.read()
        # text = " ".join(text.splitlines())
        # text = text.replace("   ", " ")
        # text = text.replace("  ", " ")
        # text = text.strip()
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
