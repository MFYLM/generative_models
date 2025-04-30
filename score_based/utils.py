from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax


@struct.dataclass
class TrainState