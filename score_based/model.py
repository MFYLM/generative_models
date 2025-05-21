import einops
import jax
import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx
from typing import List, Any, Optional, Callable


class Identity(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        self.rngs = rngs
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class GaussianFeatureTimeEmbedding(nnx.Module):
    def __init__(self, embed_dim: int, scale: float = 30,):
        super().__init__()
        key = jax.random.PRNGKey(0)
        self.w = jax.nn.initializers.normal(stddev=scale)(key, (embed_dim,))
        self.w = jax.lax.stop_gradient(self.w)
     
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_proj = x[:, None] * self.w[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class ResidualBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, time_embed_dim: int, rngs: nnx.Rngs, groups: int = 8):
        super().__init__()
        self.norm1 = nnx.GroupNorm(in_features, groups, rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            rngs=rngs
        )
        self.norm2 = nnx.GroupNorm(out_features, groups, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            rngs=rngs
        )
        self.time_mlp = nnx.Linear(time_embed_dim * 2, out_features, rngs=rngs)
        self.res_conv = nnx.Conv(in_features, out_features, kernel_size=(1, 1), rngs=rngs) \
                            if in_features != out_features else Identity(rngs=rngs)
        
    def __call__(self, x: jnp.ndarray, time_emb: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        residual = x
        
        x = self.norm1(x)
        x = jax.nn.silu(x)
        x = self.conv1(x)
        
        if time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            time_emb = nnx.silu(time_emb)            
            x = x + time_emb[:, None, None, :]
            
        x = self.norm2(x)
        x = jax.nn.silu(x)
        x = self.conv2(x)
        
        return x + self.res_conv(residual)

class Downsample(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            rngs=rngs
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x)

class Upsample(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv = nnx.ConvTranspose(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            rngs=rngs
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.conv(x)

class UNet(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        in_channels: int,
        channels: List[int],
        dropout: float = 0.1,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        channels_list = list(zip(channels[:-1], channels[1:]))
        
        self.init_conv = nnx.Conv(
            in_features=in_channels,
            out_features=channels[0],
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            rngs=rngs
        )
        
        self.time_embed = GaussianFeatureTimeEmbedding(time_embed_dim)
        
        self.down_blocks = []
        for dim_in, dim_out in channels_list:
            self.down_blocks.append([
                ResidualBlock(dim_in, dim_in, time_embed_dim, rngs=rngs),
                nnx.Dropout(dropout, rngs=rngs),
                ResidualBlock(dim_in, dim_in, time_embed_dim, rngs=rngs),
                Downsample(dim_in, dim_out, rngs=rngs)
            ])
        
        self.bottleneck = [
            ResidualBlock(channels[-1], channels[-1], time_embed_dim, rngs=rngs),
            ResidualBlock(channels[-1], channels[-1], time_embed_dim, rngs=rngs),
        ]
        
        self.up_blocks = []
        for dim_out, dim_in in channels_list[::-1]:
            self.up_blocks.append([
                Upsample(dim_in * 2, dim_out, rngs=rngs),
                ResidualBlock(dim_out, dim_out, time_embed_dim, rngs=rngs),
                nnx.Dropout(dropout, rngs=rngs),
                ResidualBlock(dim_out, dim_out, time_embed_dim, rngs=rngs),
            ])
        
        self.final_conv = nnx.Conv(
            in_features=channels[0] * 2,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            rngs=rngs
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        x: jnp.ndarray (batch_size, height, width, channels)
        t: jnp.ndarray (batch_size,)
        """
        t_emb = self.time_embed(t)
        x = self.init_conv(x)
        skips = [x]
        
        for block1, dropout, block2, downsample in self.down_blocks:
            x = block1(x, t_emb)
            x = dropout(x)
            x = block2(x, t_emb)
            x = downsample(x)
            skips.append(x)
        
        for block in self.bottleneck:
            x = block(x, t_emb)
                
        for upsample, block1, dropout, block2 in self.up_blocks:
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = upsample(x)
            x = block1(x, t_emb)
            x = dropout(x)
            x = block2(x, t_emb)
        
        x = jnp.concatenate([x, skips.pop()], axis=-1)
        return self.final_conv(x)

    
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    
    model = UNet(rngs=nnx.Rngs(rng))
    x = jnp.ones((1, 32, 32, 3))
    t = jnp.ones((1, ))
    
    x_t = model(x, t)
