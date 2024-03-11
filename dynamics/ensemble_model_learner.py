"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple, Callable

from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from dynamics.model_learner import WorldModel
from common import Batch, InfoDict, Model, PRNGKey, MLP, Params

def evaluate(model, validation_data, batch_size):
    lossR, lossD, lossM, lossT = [], [], [], []
    for idx in tqdm(range(0, validation_data.size, batch_size)):
        s = validation_data.observations[idx:idx+batch_size]
        a = validation_data.actions[idx:idx+batch_size]
        sN = validation_data.next_observations[idx:idx+batch_size]
        r = validation_data.rewards[idx:idx+batch_size, None]
        mask = validation_data.masks[idx:idx+batch_size, None]

        sN_hat, r_hat, mask_hat = model(s, a)
        mu_s, logvar_s = sN_hat
        loss_rew = (((r_hat - r[:, None]) ** 2)).mean(axis=2)
        loss_dyn = jnp.exp(-logvar_s) * ((sN[:, None] - mu_s) ** 2) + logvar_s
        loss_dyn = (mask[:, None] * loss_dyn).mean(axis=2)
        loss_mask = (mask[:, None] - mask_hat).mean(axis=2)

        loss = loss_dyn + loss_rew + loss_mask
        lossT.append(loss)
        lossD.append(loss_dyn)
        lossR.append(loss_rew)
        lossM.append(loss_mask)

    lossT = jnp.concatenate(lossT, axis=0)
    lossD = jnp.concatenate(lossD, axis=0)
    lossR = jnp.concatenate(lossR, axis=0)

    return {
        'loss': jnp.mean(lossT, axis=0),
        'lossR': jnp.mean(lossR, axis=0),
        'lossD': jnp.mean(lossD, axis=0),
    } 

def sample_step(rng, model, observations, actions):
    sN_dist, r_hat, mask_hat = model(observations, actions); means, log_vars = sN_dist
    sN_hat = means
    sN_hat, r_hat, mask_hat = sN_hat[:, 0], r_hat[:, 0], mask_hat[:, 0]
    return rng, sN_hat, r_hat, mask_hat
    #sN_dist = tfd.MultivariateNormalDiag(loc=means

@jax.jit
def _update_jit(
    rng: PRNGKey, model: Model, batch: Batch 
) -> Tuple[PRNGKey, Model, InfoDict]:

    s, a, r, sN, mask = batch.observations, batch.actions, batch.rewards[:, None], batch.next_observations, batch.masks[:, None]
    #jax.debug.print("r_hat:{r}, s_hat:{s}, 1-d:{x}", x=mask.mean(), r=r_hat.mean(), s=s_hat.mean())

    def loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        sN_hat, r_hat, mask_hat = model.apply({'params': params}, s, a)
        mu_s, logvar_s = sN_hat
        loss_rew = (((r_hat - r[:, None]) ** 2)).sum(axis=1).mean()
        #loss_rew = jnp.exp(-logvar_r) * ((r - mu_r) ** 2) + logvar_r
        loss_dyn = jnp.exp(-logvar_s) * ((sN[:, None] - mu_s) ** 2) + logvar_s
        loss_dyn = (mask[:, None] * loss_dyn).sum(axis=1).mean()
        loss_mask = ((mask[:, None] - mask_hat) ** 2).sum(axis=1).mean()
        loss = loss_dyn + loss_rew + loss_mask
        #loss = loss_rew + loss_mask

        #jax.debug.print("{x}", x = jnp.abs(r + mask).mean())
        return loss, {
            'lossR': loss_rew,
            'lossD': loss_dyn,
            'lossM': loss_mask,
        }

    new_model, info = model.apply_gradient(loss_fn)

    return rng, new_model, info

def softplus(x):
    return jnp.logaddexp(x, 0)

def soft_clamp(x, _min, _max):
    x = _max - softplus(_max - x)
    x = _min + softplus(x - _min) 
    return x

class EnsembleLinear(nn.Module):
    input_dim: int
    output_dim: int
    num_ensemble: int

    def setup(self):
        self.weight = self.param('kernel', nn.initializers.glorot_normal(), (self.num_ensemble, self.input_dim, self.output_dim))
        self.bias = self.param('bias', nn.initializers.glorot_normal(), (self.num_ensemble, 1, self.output_dim))

    def __call__(self, x: jnp.ndarray):
        x = jnp.einsum('nbi,nij->nbj', x, self.weight)
        x = x + self.bias
        return x

class EffEnsembleDynamicModel(nn.Module):
    model: nn.Module
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    reward_scaler: Tuple[float, float]
    elites: jnp.ndarray
    terminal_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    output_all: bool = False

    def __call__(self,
                 key: PRNGKey,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        z = jnp.concatenate([observations, actions], axis=1)
        z = (z - self.scaler[0]) / self.scaler[1]
        N, C = z.shape; E = self.elites.shape[0]; R = (N-1) // E + 1

        if not self.output_all:
            z = jnp.concatenate([z, jnp.zeros((E*R - N, C))], axis=0)
            idxs = jax.random.permutation(key, E*R)
            tmp = jnp.reshape(z[idxs], (E, R, C))
            z = jnp.zeros((self.model.num_models, R, C)); z = z.at[self.elites].set(tmp)

        mean, logvar = self.model(z)
        std = jnp.sqrt(jnp.exp(logvar))
        ensemble_samples = (mean + jax.random.normal(key, mean.shape) * std)

        if not self.output_all:
            Co = ensemble_samples.shape[-1]
            result = ensemble_samples[self.elites].reshape((E*R, Co))
            ensemble_samples = jnp.zeros((E*R, Co)); ensemble_samples.at[idxs].set(result)
            ensemble_samples = ensemble_samples[:N]
            samples = jnp.concatenate([ensemble_samples[:, :-1] + observations, ensemble_samples[:, -1:]], axis=1)
            next_obs = samples[..., :-1]
            reward = samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = self.terminal_fn(observations, actions, next_obs).squeeze(1)
        else:
            samples = jnp.concatenate([ensemble_samples[:, :, :-1] + observations[None], ensemble_samples[:, :, -1:]], axis=2)
            next_obs = samples[..., :-1]
            reward = samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = self.terminal_fn(observations, actions, next_obs[0]).squeeze(1)

        info = {}
        info["raw_reward"] = reward

        return next_obs, reward, terminal, info



class EnsembleDynamicModel(nn.Module):
    model: nn.Module
    scaler: Tuple[jnp.ndarray, jnp.ndarray]
    reward_scaler: Tuple[float, float]
    elites: jnp.ndarray
    terminal_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    output_all: bool = False

    def __call__(self,
                 key: PRNGKey,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        z = jnp.concatenate([observations, actions], axis=1)
        z = (z - self.scaler[0]) / self.scaler[1]
        print(z.shape)
        #observations, actions = z[:, :observations.shape[1]], z[:, observations.shape[1]:]
        mean, logvar = self.model(z)
        mean = jnp.concatenate([mean[:, :, :-1] + observations[None, :, :], mean[:, :, -1:]], axis=2)
        std = jnp.sqrt(jnp.exp(logvar))

        ensemble_samples = (mean + jax.random.normal(key, mean.shape) * std)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape

        if self.output_all:
            samples = ensemble_samples
            next_obs = samples[..., :-1]
            reward = samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = self.terminal_fn(observations, actions, next_obs[0]).squeeze(1)
        else:
            model_idxs = jax.random.choice(key, self.elites, (1, batch_size, 1))
            samples = jnp.take_along_axis(ensemble_samples, model_idxs, axis=0)[0] 
            next_obs = samples[..., :-1]
            reward = samples[..., -1] * self.reward_scaler[0] + self.reward_scaler[1]
            terminal = self.terminal_fn(observations, actions, next_obs).squeeze(1)
        info = {}
        info["raw_reward"] = reward

        return next_obs, reward, terminal, info

class EnsembleWorldModel(nn.Module):
    num_models: int
    num_elites: int
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    dropout_rate: Optional[float] = None

    def setup(self):
        hidden_dims = (self.obs_dim+self.action_dim,) + self.hidden_dims
        self.layers = [EnsembleLinear(hidden_dims[i-1], hidden_dims[i], self.num_models) for i in range(1, len(hidden_dims))]
        self.last_layer = EnsembleLinear(self.hidden_dims[-1], 2 * (self.obs_dim + 1), self.num_models)
        self.min_logvar = self.param('min_logvar', jax.nn.initializers.constant(-10), (self.obs_dim+1,))
        self.max_logvar = self.param('max_logvar', jax.nn.initializers.constant(0.5), (self.obs_dim+1,))

    def __call__(self,
                 z: jnp.ndarray,
                 #actions: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
       
        #z = jnp.concatenate([observations, actions], axis=1)
        z = z[None, :, :].repeat(self.num_models, axis=0)
        for layer in self.layers:
            z = layer(z); z = nn.swish(z)
        z = self.last_layer(z)
        #mean, logvar, reward = jnp.split(z, [self.obs_dim, 2*self.obs_dim], axis=1)
        mean, logvar = z[:, :, :self.obs_dim+1], z[:, :, self.obs_dim+1:]
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

        return mean, logvar

    def set_elites(self, metrics):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        idxs = np.sort(valid_losses)
        self.model.elites = idxs

class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 model_lr: float = 3e-4,
                 model_hidden_dims: Sequence[int] = None,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 num_models: int = 7,
                 num_elites: int = 5,
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        print(dropout_rate, model_hidden_dims, max_steps)

        rng = jax.random.PRNGKey(seed)
        rng, model_key = jax.random.split(rng, 2)

        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        model_def = EnsembleWorldModel(num_models, num_elites, model_hidden_dims, obs_dim, action_dim, dropout_rate=dropout_rate)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-model_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=model_lr)

        model = Model.create(model_def,
                             inputs=[model_key, observations, actions],
                             tx=optimiser)

        self.model = model
        self.rng = rng

    def evaluate(self, validation_data, batch_size):
        return evaluate(self.model, validation_data, batch_size)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_model, info = _update_jit(self.rng, self.model, batch)

        self.rng = new_rng
        self.model = new_model

        return info
