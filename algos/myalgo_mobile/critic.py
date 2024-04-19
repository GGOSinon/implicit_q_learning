from typing import Tuple

import jax.numpy as jnp
import jax

import numpy as np
from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss

def update_v(key: PRNGKey, critic: Model, value: Model, actor: Model, data_batch: Batch, model_batch: Batch,
             expectile: float) -> Tuple[Model, InfoDict]:

    q1_rollout, q2_rollout = critic(model_batch.observations, model_batch.actions)
    q_model = jnp.minimum(q1_rollout, q2_rollout)

    q1_data, q2_data = critic(data_batch.observations, data_batch.actions)
    q_data = jnp.minimum(q1_data, q2_data)

    q1_data_pi, q2_data_pi = critic(data_batch.observations, actor(data_batch.observations).sample(seed=key))
    q_data_pi = jnp.minimum(q1_data_pi, q2_data_pi)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v_model = value.apply({'params': value_params}, model_batch.observations)
        value_loss_model = loss(q_model, v_model, expectile)
        bellman_error_model_v = jnp.where(q_model > v_model, expectile, 1 - expectile) * (v_model - q_model)

        v_data = value.apply({'params': value_params}, data_batch.observations)
        value_loss_data = loss(q_data, v_data, 0.7)
        bellman_error_data_v = jnp.where(q_data > v_data, expectile, 1 - expectile) * (v_data - q_data)

        bellman_error_v = jnp.concatenate([bellman_error_model_v, bellman_error_data_v], axis=0)

        #value_loss = jnp.concatenate([value_loss_model, value_loss_data], axis=0).mean()
        value_loss = value_loss_model.mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_model': v_model.mean(),
            'v_data': v_data.mean(),
            'argmax_adv': (q_data_pi - q_data).mean(),
            'bellman_error_model_v': bellman_error_model_v.mean(),
            'bellman_error_data_v': bellman_error_data_v.mean(),
            'bellman_error_v': bellman_error_v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info

# COMBO
def update_q(key: PRNGKey, critic: Model, target_critic: Model, actor: Model, model: Model,
             data_batch: Batch, model_batch: Batch, model_batch_ratio: float,
             discount: float, temperature: float, H: int, expectile: float, num_repeat: int) -> Tuple[Model, Model, InfoDict]:

    key1, key2, key3, key4 = jax.random.split(key, 4)
 
    N = model_batch.observations.shape[0]
    Robs = model_batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    #Ra = model_batch.actions[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    Ra = actor(Robs, temperature).sample(seed=key1)
    states, rewards, actions, masks = [Robs], [], [Ra], []
    for i in range(H):
        s, a = states[-1], actions[-1]
        rng1, rng2, key1 = jax.random.split(key1, 3)
        s_next, rew, terminal, _ = model(rng2, s, a)
        a_next = actor(s_next, temperature).sample(seed=rng1)
        states.append(s_next)
        actions.append(a_next)
        rewards.append(rew)
        masks.append(1 - terminal)
    
    target_q_rollout, loss_weights = [], [jnp.ones_like(rewards[i])]
    for i in range(H):
        loss_weights.append(loss_weights[-1] * discount * masks[i])
        next_value = critic(states[i+1], actions[i+1])
        #target_q_rollout.append(rewards[i] + discount * masks[i] * next_value)

    #target_q_rollout = target_q_rollout[0]
    #states = states[0]
    #actions = actions[0]
    #loss_weights = loss_weights[0]

    target_q_rollout = jnp.concatenate(target_q_rollout, axis=0)
    states = jnp.concatenate(states[:-1], axis=0)
    actions = jnp.concatenate(actions[:-1], axis=0)
    rewards = jnp.concatenate(rewards, axis=0)
    loss_weights = jnp.concatenate(loss_weights[:-1], axis=0)

    next_a = actor(data_batch.next_observations, temperature).sample(seed=key1)
    next_value = critic(data_batch.next_observations, next_a)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_value

    ###### CQL ######

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        
        ## Data critic loss (unpenalized)
        q_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        critic_loss_data = loss(target_q_data, q_data, 0.5)

        ## Rollout critic loss (penalized)
        q_rollout = critic.apply({'params': critic_params}, states, actions)
        critic_loss_rollout = loss(target_q_rollout, q_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).reshape(-1, N, num_repeat).mean(axis=(0, 2))

        critic_loss = critic_loss_data.mean() * (1 - model_batch_ratio) + critic_loss_rollout.mean() * model_batch_ratio

        ## Target network EMA loss
        q_target_data = target_critic(data_batch.observations, data_batch.actions)
        critic_reg_loss_data = (q_target_data - q_data) ** 2
        q_target_rollout = target_critic(states, actions)
        critic_reg_loss_rollout = (q_target_rollout - q_rollout) ** 2
        critic_reg_loss_rollout = critic_reg_loss_rollout * loss_weights
        critic_reg_loss = critic_reg_loss_data.mean() * (1 - model_batch_ratio) + critic_reg_loss_rollout.mean() * model_batch_ratio

        critic_info = {
            'critic_loss': critic_loss, 'critic_loss_data': critic_loss_data.mean(), 'critic_loss_model': critic_loss_rollout.mean(), 'loss_weights': loss_weights.mean(),
            'q_data': q_data.mean(), 'q_data_min': q_data.min(), 'q_data_max': q_data.max(), 
            'q_rollout': q_rollout.mean(), 'q_rollout_min': q_rollout.min(), 'q_rollout_max': q_rollout.max(), 'q_adv': q_rollout.mean() - q_data.mean(),
            'reward_data': data_batch.rewards.mean(), 'reward_data_max': data_batch.rewards.max(), 'reward_data_min': data_batch.rewards.min(),
            'reward_model': (rewards * loss_weights).mean(), 'reward_max': (rewards * loss_weights).max(), 'reward_min': (rewards * loss_weights).min(),
            'expectile': expectile, 'state_max': jnp.abs(states * loss_weights[:, None]).max(), 'critic_reg_loss': critic_reg_loss,
        } 
        return critic_loss + critic_reg_loss, critic_info 

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, {**info}

def update_tau_model(tau_model:Model, bellman_error: jnp.ndarray):
    def tau_model_loss_fn(tau_model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_expectile = tau_model.apply({'params': tau_model_params}); expectile = jax.nn.sigmoid(log_expectile)
        tau_model_loss = -(log_expectile * bellman_error)
        return tau_model_loss, {}
    new_tau_model, info = tau_model.apply_gradient(tau_model_loss_fn)
    return new_tau_model, info

