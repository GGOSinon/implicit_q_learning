"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import gym
import os
import torch

import policy
import value_net
from algos.myalgo_mopo.actor import reinforce_update_actor, awr_update_actor, update_actor, gae_update_actor, update_alpha, update_all
from common import Batch, InfoDict, Model, PRNGKey, inverse_sigmoid
#from algos.myalgo_nstep.critic import update_q, update_v, update_tau_model
from algos.myalgo_mopo.critic import update_q, update_v, update_tau_model, update_q_baseline
from algos.iql.critic import update_q as update_iql_q, update_v as update_iql_v

from dynamics.termination_fns import get_termination_fn
from dynamics.ensemble_model_learner import EnsembleWorldModel, sample_step, EnsembleDynamicModel
from dynamics.model_learner import WorldModel
from dynamics.oracle import MujocoOracleDynamics
from offlinerlkit.dynamics.ensemble_dynamics import EnsembleDynamics
from offlinerlkit.modules import EnsembleDynamicsModel
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn as get_termination_fn_torch
from functools import partial

def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=['max_q_backup', 'horizon_length', 'num_actor_updates', 'baseline'])
def _update_jit(
        rng: PRNGKey, actor: Model, baseline_actor: Model, sac_alpha: Model, critic: Model, baseline_critic: Model, baseline_value: Model, target_critic: Model, target_baseline_critic: Model, model: Model, tau_model: Model,
        data_batch: Batch, model_batch: Batch, model_batch_ratio: float, discount: float, tau: float, 
        expectile: float, expectile_policy: float, temperature: float, target_entropy: float, lamb: float, horizon_length: int, num_actor_updates: int, baseline: str
    ) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, InfoDict]:
    
    log_alpha = sac_alpha(); alpha = jnp.exp(log_alpha)
    #log_expectile = tau_model(); expectile = jax.nn.sigmoid(log_expectile)
    #alpha = 0.1
    mix_batch = Batch(observations=jnp.concatenate([data_batch.observations, model_batch.observations], axis=0),
                      actions=jnp.concatenate([data_batch.actions, model_batch.actions], axis=0),
                      rewards=jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0),
                      masks=jnp.concatenate([data_batch.masks, model_batch.masks], axis=0),
                      next_observations=jnp.concatenate([data_batch.next_observations, model_batch.next_observations], axis=0),
                      returns_to_go=None,)

    #new_value, value_info = update_v(rng, target_critic, value, actor, data_batch, model_batch, expectile_policy)
    key, key2, key3, rng = jax.random.split(rng, 4)

    if True:
        new_actor = actor
        for _ in range(num_actor_updates):
            #new_actor, actor_info = gae_update_actor(key, new_actor, critic, model,
            #                                     model_batch, discount, temperature, alpha, lamb, horizon_length, expectile)
            new_actor, actor_info = update_actor(key, new_actor, critic, model,
                                                 mix_batch, discount, temperature, alpha)
        #new_actor, actor_info = awr_update_actor(key, actor, target_critic, new_value, model,
        #                                         model_batch, discount, temperature, alpha)
        #new_actor, actor_info = reinforce_update_actor(key, actor, target_critic, new_value, model,
        #                                         model_batch, discount, temperature, alpha)
        new_alpha, alpha_info = update_alpha(key2, actor, sac_alpha, mix_batch, target_entropy)

        new_critic, critic_info = update_q(key3, critic, target_critic, new_actor, model,
                                           data_batch, model_batch, model_batch_ratio,
                                           discount, lamb, horizon_length, expectile, target_baseline_critic) 
    
    else:
        new_actor, new_critic, new_alpha, actor_info, critic_info, alpha_info = update_all(
            key3, actor, critic, target_critic, model, sac_alpha, target_baseline_critic,
            data_batch, model_batch, model_batch_ratio,
            discount, lamb, horizon_length, expectile, target_entropy,
        ) 

    if baseline is None:
        new_baseline_critic = baseline_critic
        baseline_info = {}

    if baseline == 'random':
        new_baseline_critic, baseline_critic_info = update_q_baseline(key, baseline_critic, target_baseline_critic, baseline_actor, model, model_batch, discount, 0.5)
        new_baseline_value = baseline_value
        baseline_info = {**baseline_critic_info}
        new_target_baseline_critic = target_update(baseline_critic, target_baseline_critic, tau)
    
    if baseline == 'iql':
        new_baseline_critic, baseline_critic_info = update_iql_q(baseline_critic, baseline_value, data_batch, discount)
        new_baseline_value, baseline_value_info = update_iql_v(target_baseline_critic, baseline_value, data_batch, 0.5)
        baseline_info = {**baseline_critic_info, **baseline_value_info}
        baseline_info = {'iql_'+k: v for (k, v) in baseline_info.items()}
        new_target_baseline_critic = target_update(baseline_critic, target_baseline_critic, tau)

    new_tau_model, tau_model_info = update_tau_model(tau_model, 0.)
    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_alpha, new_tau_model, new_critic, new_baseline_critic, new_baseline_value, new_target_critic, new_target_baseline_critic, {
        **critic_info,
        **baseline_info,
        #**value_info,
        **actor_info,
        **alpha_info,
        **tau_model_info,
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
                 expectile: float = 0.1,
                 expectile_policy: float = 0.1,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 num_models: int = 7,
                 num_elites: int = 5,
                 model_hidden_dims: Sequence[int] = (256, 256, 256),
                 lamb: float = 0.95,
                 horizon_length: int = None,
                 reward_scaler: Tuple[float, float] = None,
                 num_actor_updates: int = None,
                 baseline: str = None,
                 #sac_alpha: float = 0.2,
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        obs_dim = observations.shape[-1] 
        action_dim = actions.shape[-1]

        self.expectile = expectile
        self.expectile_policy = expectile_policy
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.target_entropy = -action_dim
        self.horizon_length = horizon_length
        self.lamb = lamb
        self.num_actor_updates = num_actor_updates
        self.baseline = baseline
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
        if self.dynamics == 'dreamer':
            import dreamerv3
            import dreamerv3.embodied
            from dreamerv3.embodied.envs import from_gym
            import dreamerv3.ninjax as nj
            config = dreamerv3.embodied.Config(dreamerv3.configs['defaults'])
            config = config.update(dreamerv3.configs['medium'])
            config = config.update({
                'logdir': f'./logdir/{env_name}',
                'encoder.mlp_keys': 'vector',
                'decoder.mlp_keys': 'vector',
            })
            logdir = dreamerv3.embodied.Path(config.logdir) 
            #config = embodied.Flags(config).parse()

            env = gym.make(env_name)
            env = from_gym.FromGym(env, obs_key='vector')
            env = dreamerv3.wrap_env(env, config)
            env = dreamerv3.embodied.BatchEnv([env], parallel=False)

            step = dreamerv3.embodied.Counter()
            replay = dreamerv3.embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'replay')

            ckpt = dreamerv3.embodied.Checkpoint() 
            ckpt.step = step
            ckpt.agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
            ckpt.replay = replay
            path = '~/logdir/run1/checkpoint.ckpt' 
            ckpt.load(path)
            print(ckpt._values.keys())

            model = ckpt._values['agent'].agent.wm
            print(model)
            
            scaler = obs_scaler = (0., 1.)
            jax.config.update("jax_transfer_guard", "allow")

            self.model = model

            def step_dreamer(observations, actions):
                prev_latent, prev_action = model.initial(observations.shape[0])
                next_latent = self.model.rssm.img_step(prev_latent, actions)
                rewards = self.model.heads['reward'](next_latent).mode()
                conts = self.model.heads['cont'](next_latent).mode()
                next_obs = self.model.heads['decoder'](next_latent)['vector'].mode()
                return next_obs, rewards, conts

            self.init_fn = jax.jit(nj.pure(lambda x: model.initial(x.shape[0])))
            self.step_fn = jax.jit(nj.pure(model.rssm.img_step))
            self.step_dreamer = jax.jit(nj.pure(step_dreamer))

            key = jax.random.PRNGKey(42)
            latent, self.model_state = self.init_fn({}, key, observations)

            #parsed, other = dreamerv3.embodied.Flags(configs=['defaults']).parse_known(None)
            #config = dreamerv3.embodied.Config(dreamerv3.agent.Agent.configs['defaults'])
            #model = DreamerV3WorldModel({'vector': obs_space}, {'action': act_space}, config, name='wm')

        if self.dynamics == 'torch':
            mu = np.load(os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', env_name, 'mu.npy'))
            std = np.load(os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', env_name, 'std.npy'))
            scaler = (jnp.array(mu), jnp.array(std))
            obs_scaler = (jnp.array(mu[:, :obs_dim]), jnp.array(std[:, :obs_dim]))

            ckpt = torch.load(os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', env_name, 'dynamics.pth'))
            ckpt = {k: v.cpu().numpy() for (k, v) in ckpt.items()}
            elites = ckpt['elites']

            self.termination_fn = get_termination_fn(task=env_name)
            model_def = EnsembleWorldModel(num_models, num_elites, model_hidden_dims, obs_dim, action_dim, dropout_rate=None)
            model_def = EnsembleDynamicModel(model_def, scaler, reward_scaler, elites, self.termination_fn)
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

        actor_def = policy.NormalTanhPolicy(obs_scaler,
                                            hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=True,
                                            tanh_squash_distribution=True)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
            baseline_actor_optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_optimiser = optax.adam(learning_rate=actor_lr)
            baseline_actor_optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=actor_optimiser)

        alpha_def = policy.SACalpha() 
        alpha = Model.create(alpha_def, inputs=[alpha_key], tx=optax.adam(learning_rate=alpha_lr))

        tau_model_def = policy.SACalpha(init_value = inverse_sigmoid(expectile)) 
        tau_model = Model.create(tau_model_def, inputs=[alpha_key], tx=optax.adam(learning_rate=1e-2))

        critic_def = value_net.DoubleCritic(scaler, hidden_dims)
        critic_opt = optax.adam(learning_rate=value_lr)
        critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx = critic_opt)

        value_def = value_net.ValueCritic(obs_scaler, hidden_dims)
        value_opt = optax.adam(learning_rate=value_lr)
        value = Model.create(value_def, inputs=[value_key, observations], tx=value_opt)

        target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
        target_value = Model.create(value_def, inputs=[value_key, observations])

        if baseline is not None:
            baseline_actor = Model.create(actor_def, inputs=[actor_key, observations], tx=baseline_actor_optimiser)

            baseline_critic_opt = optax.adam(learning_rate=value_lr)
            baseline_critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx = baseline_critic_opt)

            baseline_value_opt = optax.adam(learning_rate=value_lr)
            baseline_value = Model.create(value_def, inputs=[value_key, observations], tx = baseline_value_opt)

            target_baseline_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
        else:
            baseline_actor = None
            baseline_critic = None
            baseline_value = None
            target_baseline_critic = None

        self.actor = actor
        self.baseline_actor = baseline_actor
        self.alpha = alpha
        self.critic = critic
        self.baseline_critic = baseline_critic
        self.value = value
        self.baseline_value = baseline_value
        self.model = model
        self.tau_model = tau_model
        self.target_critic = target_critic
        self.target_baseline_critic = target_baseline_critic
        self.target_value = target_value
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor, observations, temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def step(self,
             key: PRNGKey,
             observations: np.ndarray,
             actions: np.ndarray) -> np.ndarray:
        #rng, sN, rewards, masks = sample_step(self.rng, self.model, observations, actions)
        #self.rng = rng
        if self.dynamics == 'torch':
            next_obs, rewards, terminals, _ = self.model(key, observations, actions)
        if self.dynamics == 'dreamer':
            key = jax.random.PRNGKey(42)
            next_obs, rewards, terminals = self.step_dreamer(self.model_state, key, observations, actions)[0]
            print(next_obs, rewards, terminals)
        rewards, masks = rewards, 1- terminals

        return next_obs, rewards, masks

    def rollout(self,
                key: PRNGKey,
                observations: np.ndarray,
                rollout_length: int,
                temperature: float=1.0,) -> np.ndarray:
        states, actions, rewards, masks = [observations], [], [], []
        
        for _ in range(rollout_length):
            key, rng = jax.random.split(key)
            action = self.sample_actions(states[-1], temperature)
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
        

    def update(self, data_batch: Batch, model_batch: Batch, model_batch_ratio: float) -> InfoDict:
        new_rng, new_actor, new_alpha, new_tau_model, new_critic, new_baseline_critic, new_baseline_value, new_target_critic, new_target_baseline_critic, info = _update_jit(
            self.rng, self.actor, self.baseline_actor, self.alpha, self.critic, self.baseline_critic, self.baseline_value, self.target_critic, self.target_baseline_critic, self.model, self.tau_model,
            data_batch, model_batch, model_batch_ratio, self.discount, self.tau,
            self.expectile, self.expectile_policy, self.temperature, self.target_entropy, self.lamb, self.horizon_length, self.num_actor_updates, self.baseline)

        self.rng = new_rng
        self.actor = new_actor
        self.alpha = new_alpha
        self.tau_model = new_tau_model
        self.critic = new_critic
        self.baseline_critic = new_baseline_critic
        self.baseline_value = new_baseline_value
        self.target_critic = new_target_critic
        self.target_baseline_critic = new_target_baseline_critic

        return info
