from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax
from typing import List


@struct.dataclass
class TrainState(struct.PyTreeNode):
    step: int
    optimizer: optax.Optimizer
    model: nnx.Module
    rngs: nnx.Rngs


@struct.dataclass
class TrainConfig:
    learning_rate: float
    lr_schedule: optax.Schedule
    batch_size: int
    num_epochs: int
    num_steps: int
    sigma: float
    channels: List[int]
    embed_dim: int
    dropout: float
    time_embed_dim: int
    device: str

