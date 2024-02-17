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
def update_q(key: PRNGKey, critic: Model, target_value: Model, actor: Model, model: Model,
             data_batch: Batch, model_batch: Batch, discount: float, 
             lamb: float, H: int, expectile: float) -> Tuple[Model, Model, InfoDict]:

    key1, key2, key3, key4 = jax.random.split(key, 4)
 
    num_repeat = 50; N = model_batch.observations.shape[0]
    Robs = model_batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    #Ra = model_batch.actions[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    Ra = actor(Robs).sample(seed=key1)
    states, rewards, actions, masks = [Robs], [], [Ra], []
    for i in range(H):
        s, a = states[-1], actions[-1]
        rng1, rng2, key1 = jax.random.split(key1, 3)
        s_next, rew, terminal, _ = model(rng2, s, a)
        a_next = actor(s_next).sample(seed=rng1)
        states.append(s_next)
        actions.append(a_next)
        rewards.append(rew)
        masks.append(1 - terminal)

    next_value = target_value(states[-1])
    target_q_rollout = []
    value_estimate = next_value
    for i in reversed(range(H)):
        next_value = target_value(states[i+1])
        value_estimate = rewards[i] + discount * masks[i] * (lamb * value_estimate + (1-lamb) * next_value)
        target_q_rollout.append(value_estimate)

    loss_weights = [jnp.ones_like(rewards[i])]
    for i in range(H):
        loss_weights.append(loss_weights[-1] * lamb * masks[i])
    target_q_rollout = list(reversed(target_q_rollout))

    target_q_rollout = jnp.concatenate(target_q_rollout, axis=0)
    states = jnp.concatenate(states[:-1], axis=0)
    actions = jnp.concatenate(actions[:-1], axis=0)
    loss_weights = jnp.concatenate(loss_weights[:-1], axis=0)

    #target_q_rollout = target_q_rollout.reshape(N, num_repeat)
    #target_q_rollout = jnp.take_along_axis(target_q_rollout, jnp.argmin(target_q_rollout, axis=1)[:, None], axis=1).squeeze(1)

    next_a = actor(data_batch.next_observations).sample(seed=key1)
    next_value = target_value(data_batch.next_observations)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_value

    #target_q_data = jnp.maximum(target_q_data, -50.)
    #target_q_rollout = jnp.maximum(target_q_rollout, -50.)

    #target_q_data = jnp.maximum(target_q_data, data_batch.returns_to_go)

    #rollout_ratio = (target_q_rollout[-1] > target_q_td).mean()
    #target_q = jnp.maximum(target_q_rollout, target_q_td)
    #target_q = target_q_rollout

    ###### CQL ######

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_data, q2_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        critic_loss_data = loss(target_q_data, q1_data, 0.5) + loss(target_q_data, q2_data, 0.5)
        bellman_error_data_q1 = jnp.where(target_q_data > q1_data, 0.5, 0.5) * (q1_data - target_q_data)
        bellman_error_data_q2 = jnp.where(target_q_data > q2_data, 0.5, 0.5) * (q2_data - target_q_data)
        bellman_error_data = bellman_error_data_q1 + bellman_error_data_q2

        q1_rollout, q2_rollout = critic.apply({'params': critic_params}, states, actions)
        critic_loss_rollout = loss(target_q_rollout, q1_rollout, expectile) + loss(target_q_rollout, q2_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).reshape(H, N, num_repeat).mean(axis=(0, 2))
        bellman_error_rollout_q1 = jnp.where(target_q_rollout > q1_rollout, expectile, 1-expectile) * (q1_rollout - target_q_rollout)
        bellman_error_rollout_q2 = jnp.where(target_q_rollout > q2_rollout, expectile, 1-expectile) * (q2_rollout - target_q_rollout)
        bellman_error_rollout = bellman_error_rollout_q1 + bellman_error_rollout_q2

        bellman_error = jnp.concatenate([bellman_error_data, bellman_error_rollout], axis=0)

        critic_loss = jnp.concatenate([critic_loss_data, critic_loss_rollout], axis=0)
        critic_loss = critic_loss.mean()

        return critic_loss, {
            'critic_loss': critic_loss, 'loss_weights': loss_weights.mean(), 'returns_to_go': data_batch.returns_to_go.mean(),
            'q1_data': q1_data.mean(), 'q1_data_min': q1_data.min(), 'q1_data_max': q1_data.max(),  
            'q2_data': q2_data.mean(), 'q2_data_min': q2_data.min(), 'q2_data_max': q2_data.max(),
            'q1_rollout': q1_rollout.mean(), 'q1_rollout_min': q1_rollout.min(), 'q1_rollout_max': q1_rollout.max(), 'q1_adv': q1_rollout.mean() - q1_data.mean(),
            'q2_rollout': q2_rollout.mean(), 'q2_rollout_min': q2_rollout.min(), 'q2_rollout_max': q2_rollout.max(), 'q2_adv': q2_rollout.mean() - q2_data.mean(),
            'reward_data': data_batch.rewards.mean(), 'reward_data_max': data_batch.rewards.max(), 'reward_data_min': data_batch.rewards.min(),
            'reward_model': jnp.concatenate(rewards, axis=0).mean(), 'reward_max': jnp.concatenate(rewards, axis=0).max(),'reward_min': jnp.concatenate(rewards, axis=0).min(),
            'bellman_error_data': bellman_error_data.mean(), 'bellman_error_rollout': bellman_error_rollout.mean(), 'bellman_error': bellman_error.mean(), 'expectile': expectile,
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, {**info}

def update_tau_model(tau_model:Model, bellman_error: jnp.ndarray):
    def tau_model_loss_fn(tau_model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_expectile = tau_model.apply({'params': tau_model_params}); expectile = jax.nn.sigmoid(log_expectile)
        tau_model_loss = -(log_expectile * bellman_error)
        return tau_model_loss, {}
    new_tau_model, info = tau_model.apply_gradient(tau_model_loss_fn)
    return new_tau_model, info

