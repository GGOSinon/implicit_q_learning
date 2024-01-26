"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import gym

import policy
import value_net
from algos.myalgo.actor import update_actor, update_alpha
from common import Batch, InfoDict, Model, PRNGKey
from algos.myalgo.critic import update_q, update_v

from dynamics.termination_fns import get_termination_fn
from dynamics.ensemble_model_learner import EnsembleWorldModel, sample_step, EnsembleDynamicModel
from dynamics.model_learner import WorldModel
from dynamics.oracle import MujocoOracleDynamics
from offlinerlkit.dynamics.ensemble_dynamics import EnsembleDynamics
from offlinerlkit.modules import EnsembleDynamicsModel
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn as get_termination_fn_torch

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
        rng: PRNGKey, actor: Model, sac_alpha: Model, critic: Model, value: Model, target_critic: Model, cql_beta: Model, model: Model,
        data_batch: Batch, model_batch: Batch, discount: float, tau: float,
        expectile: float, temperature: float, cql_weight: float, target_entropy: float, target_beta: float,
    ) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, InfoDict]:

    log_alpha = sac_alpha(); alpha = jnp.exp(log_alpha)
    mix_batch = Batch(observations=jnp.concatenate([data_batch.observations, model_batch.observations], axis=0),
                      actions=jnp.concatenate([data_batch.actions, model_batch.actions], axis=0),
                      rewards=jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0),
                      masks=jnp.concatenate([data_batch.masks, model_batch.masks], axis=0),
                      next_observations=jnp.concatenate([data_batch.next_observations, model_batch.next_observations], axis=0),)

    new_value, value_info = update_v(target_critic, value, mix_batch, expectile)
    key, key2, key3, rng = jax.random.split(rng, 4)
    new_actor, actor_info = update_actor(key, actor, target_critic,
                                             new_value, mix_batch, temperature, alpha)
    new_alpha, alpha_info = update_alpha(key2, actor, sac_alpha, mix_batch, target_entropy)

    #new_critic, critic_info = update_q(key2, critic, new_value, model, actor, batch, discount, lamb=1.0, H=5)
    new_critic, new_cql_beta, critic_info = update_q(key3, critic, target_critic, actor, cql_beta, model,
                                                     data_batch, model_batch, discount, cql_weight, target_beta)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_alpha, new_critic, new_value, new_target_critic, new_cql_beta, {
        **critic_info,
        **value_info,
        **actor_info,
        **alpha_info,
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 dynamics_name: str = None,
                 env_name: str = None,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 1e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 num_models: int = 7,
                 num_elites: int = 5,
                 model_hidden_dims: Sequence[int] = (256, 256, 256),
                 cql_weight: float = None,
                 target_beta: float = None,
                 #sac_alpha: float = 0.2,
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        obs_dim = observations.shape[-1] 
        action_dim = actions.shape[-1]

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.cql_weight = cql_weight
        self.target_entropy = -action_dim
        self.target_beta = target_beta
        #self.sac_alpha = sac_alpha

        rng = jax.random.PRNGKey(seed)
        rng, model_key, actor_key, alpha_key, critic_key, value_key = jax.random.split(rng, 6)

        self.dynamics = dynamics_name
        if self.dynamics == 'ensemble':
            model_def = EnsembleWorldModel(num_models, num_elites, model_hidden_dims, obs_dim, action_dim, dropout_rate=None)
            model = Model.create(model_def, inputs=[model_key, observations, actions], tx=None)
        if self.dynamics == 'single':
            model_def = WorldModel(model_hidden_dims, obs_dim, action_dim, dropout_rate=None)
            model = Model.create(model_def, inputs=[model_key, observations, actions], tx=None)
        if self.dynamics == 'oracle':
            model = MujocoOracleDynamics(env)
        if self.dynamics == 'torch':
            mu = np.load(os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', FLAGS.env_name, 'mu.npy'))
            std = np.load(os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', FLAGS.env_name, 'std.npy'))
            scaler = (mu, std)

            ckpt = torch.load(os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', FLAGS.env_name, 'dynamics.pth'))
            ckpt = {k: v.cpu().numpy() for (k, v) in ckpt.items()}
            elites = ckpt['elites']

            termination_fn = get_termination_fn(task=env_name)
            model_def = EnsembleWorldModel(num_models, num_elites, model_hidden_dims, obs_dim, action_dim, dropout_rate=None)
            model_def = EnsembleDynamicModel(model_def, scaler, elites, termination_fn)
            model = Model.create(model_def, inputs=[model_key, model_key, observations, actions], tx=None)

            ckpt_jax = {}
            for i in range(4):
                ckpt_jax[f'layers_{i}'] = {}
                ckpt_jax[f'layers_{i}']['kernel'] = ckpt[f'backbones.{i}.weight']
                ckpt_jax[f'layers_{i}']['bias'] = ckpt[f'backbones.{i}.bias']
            ckpt_jax[f'last_layer'] = {}
            ckpt_jax[f'last_layer']['kernel'] = ckpt[f'output_layer.weight']
            ckpt_jax[f'last_layer']['bias'] = ckpt[f'output_layer.bias']
            ckpt_jax['min_logvar'] = ckpt['min_logvar']
            ckpt_jax['max_logvar'] = ckpt['max_logvar']
            ckpt_jaxs = {'model': ckpt_jax}
            model = model.replace(params = ckpt_jaxs)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=True)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        alpha_def = policy.SACalpha()
        alpha = Model.create(alpha_def,
                             inputs=[alpha_key],
                             tx=optax.adam(learning_rate=alpha_lr))

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        if self.target_beta is None:
            self.cql_beta = None
        else:
            beta_def = policy.SACalpha()
            self.cql_beta = Model.create(beta_def,
                                         inputs=[alpha_key],
                                         tx=optax.adam(learning_rate=alpha_lr))

        self.actor = actor
        self.alpha = alpha
        self.critic = critic
        self.value = value
        self.model = model
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def step(self,
             key: PRNGKey,
             observations: np.ndarray,
             actions: np.ndarray) -> np.ndarray:
        #rng, sN, rewards, masks = sample_step(self.rng, self.model, observations, actions)
        #self.rng = rng
        sN, rewards, terminals, _ = self.model(key, observations, actions)
        rewards, masks = rewards[:, 0], 1- terminals[:, 0]

        return sN, rewards, masks

    def rollout(self,
                key: PRNGKey,
                observations: np.ndarray,
                rollout_length: int) -> np.ndarray:
        states, actions, rewards, masks = [observations], [], [], []
        
        for _ in range(rollout_length):
            key, rng = jax.random.split(key)
            action = self.sample_actions(states[-1])
            next_obs, reward, mask = self.step(rng, states[-1], action)
            states.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            masks.append(mask)

        obss = np.concatenate(states[:-1], axis=0)
        next_obss = np.concatenate(states[1:], axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        masks = np.concatenate(masks, axis=0)

        return {'obss': obss, 'actions': actions, 'rewards': rewards, 'masks': masks, 'next_obss': next_obss}
        

    def update(self, data_batch: Batch, model_batch: Batch) -> InfoDict:
        new_rng, new_actor, new_alpha, new_critic, new_value, new_target_critic, new_cql_beta, info = _update_jit(
            self.rng, self.actor, self.alpha, self.critic, self.value, self.target_critic, self.cql_beta, self.model,
            data_batch, model_batch, self.discount, self.tau, self.expectile, self.temperature, self.cql_weight, self.target_entropy, self.target_beta)

        self.rng = new_rng
        self.actor = new_actor
        self.alpha = new_alpha
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic
        self.cql_beta = new_cql_beta

        return info
