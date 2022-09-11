from jaxtyping import Array
from typing import *

import jax
import jax.numpy as jnp

from einops import reduce


def cross_entropy(
    logits: Array,  # Unnormalized predicted scores
    targets: Array,  # Ground-truth distribution (`p`)
) -> Array:
    """
    Returns the cross-entropy between the predicted distribution derived from
    the unnormalized scores/logits and the target distribution.

    Args:
        logits: The unnormalized predicted scores from your model of shape:
            [..., num classes]
        targets: The one-hot encoded targets of shape: [..., num_classes]
    """
    log_preds = jax.nn.log_softmax(logits, axis=-1)  # `log(q)`
    losses = -jnp.sum(targets * log_preds)
    return reduce(losses, "b ... -> b", "mean")


def mse(
    inputs: Array,
    targets: Array,
) -> Array:
    l2_error = (targets - inputs) ** 2
    return reduce(l2_error, "b ... -> b", "mean")


def test_mse():
    x = jnp.ones((2, 3, 32, 32))
    y = jnp.ones((2, 3, 32, 32))
    return mse(x, y)

if __name__ == "__main__":
    test_mse()
