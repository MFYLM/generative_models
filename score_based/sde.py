import jax
import jax.numpy as jnp
from flax import nnx
from model import UNet
from typing import List
from generative_models.score_based.utils import TrainState


class ScoreSDE:
    def __init__(
        self, 
        rngs: nnx.Rings,
        sigma: float,
        in_channels: int = 3,
        channels: List[int] = [256, 128, 64, 32],
        embed_dim: int = 256,
    ):
        self.sigma = sigma
        self.score_net = UNet(rngs=rngs, in_channels=in_channels, channels=channels, time_embed_dim=embed_dim)
        
    
    def marginal_prob_std(self, t, sigma):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

        Returns:
        The standard deviation.
        """
        return jnp.sqrt((sigma**(2 * t) - 1.) / 2. / jnp.log(sigma))
    
    def diffusion_coeff(self, t, sigma):
        """Compute the diffusion coefficient of our SDE.

        Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

        Returns:
        The vector of diffusion coefficients.
        """
        return sigma**t
    
    def compute_loss(self, x: jnp.ndarray, rngs: nnx.Rngs):
        random_t = jax.random.normal(rngs.next(), x.shape[0])
        z = jax.random.normal(rngs.next(), x.shape)
        std = self.marginal_prob_std(random_t)
        x_t = x + z * std[:, None, None, None]
        score = self.score_net(x_t, random_t)
        loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z)**2, axis=(1, 2, 3)))
        return loss
    
    def update(self, train_state: TrainState, x: jnp.ndarray, rngs: nnx.Rngs):
        grads = jax.grad(self.compute_loss)(train_state.model, x, rngs)
        train_state = train_state.apply_gradients(grads)
        return train_state
    
    def sample(self, x: jnp.ndarray, t: jnp.ndarray, rngs: nnx.Rngs):
        std = self.marginal_prob_std(t)
        diff_coef = self.diffusion_coeff(t, self.sigma)
        z = jax.random.normal(rngs.next(), x.shape)
        x_t = x + z * std[:, None, None, None]
        score = self.score_net(x_t, t)
        x_t = x_t - diff_coef[:, None, None, None] * score
        return x_t
        
