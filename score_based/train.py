import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
from generative_models.score_based.utils import TrainState, TrainConfig
from shared import array_typing as at
from generative_models.score_based.sde import ScoreSDE


class ScoreNetTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.rngs = nnx.Rngs(jax.random.PRNGKey(0))
        self.optimizer = optax.adamw(learning_rate=config.learning_rate)
        self.model = ScoreSDE(self.rngs, config.sigma)
        self.train_state = TrainState(step=0, optimizer=self.optimizer, model=self.model, rngs=self.rngs)
    