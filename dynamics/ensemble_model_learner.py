"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

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

class EnsembleWorldModel(nn.Module):
    num_models: int
    num_elites: int
    hidden_dims: Sequence[int]
    obs_dim: int
    action_dim: int
    dropout_rate: Optional[float] = None

    def setup(self):
        self.models = [WorldModel(self.hidden_dims, self.obs_dim, self.action_dim, self.dropout_rate) for _ in range(self.num_models)]
        self.elites = jnp.ndarray()

    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 indexs: Sequence[int] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
       
        if indexs is None:
            indexs = list(range(self.num_models))
        #inp = jnp.concatenate([observations, actions], axis = 1)
        mu_hats, logvar_hats, r_hats, mask_hats = [], [], [], []
        for idx in indexs:
            model = self.models[idx]
            sN_hat, r_hat, mask_hat = model(observations, actions, training=training); mu, logvar = sN_hat
            mu_hats.append(mu)
            logvar_hats.append(logvar)
            r_hats.append(r_hat)
            mask_hats.append(mask_hat)
        mu_hats = jnp.stack(mu_hats, axis=1)
        logvar_hats = jnp.stack(logvar_hats, axis=1)
        r_hats = jnp.stack(r_hats, axis=1)
        mask_hats = jnp.stack(mask_hats, axis=1)

        return (mu_hats, logvar_hats), r_hats, mask_hats

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

