from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    hidden_dims: Sequence[int]
    use_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        mu = self.scaler[0].reshape((1,) * (len(observations.shape) - 1) + (self.scaler[0].shape[-1],))
        std = self.scaler[1].reshape((1,) * (len(observations.shape) - 1) + (self.scaler[1].shape[-1],))
        observations = (observations - mu) / std
        critic = MLP((*self.hidden_dims, 1), use_norm=self.use_norm, use_symlog=True)(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        mu = self.scaler[0].reshape((1,) * (len(inputs.shape) - 1) + (self.scaler[0].shape[-1],))
        std = self.scaler[1].reshape((1,) * (len(inputs.shape) - 1) + (self.scaler[1].shape[-1],))
        inputs = (inputs - mu) / std
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations, use_norm=self.use_norm, use_symlog=True)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.scaler, self.hidden_dims,
                         activations=self.activations, use_norm=self.use_norm)(observations, actions)
        critic2 = Critic(self.scaler, self.hidden_dims,
                         activations=self.activations, use_norm=self.use_norm)(observations, actions)
        return critic1, critic1
