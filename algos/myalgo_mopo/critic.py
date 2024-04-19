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
def update_q_baseline(key: PRNGKey, critic: Model, target_critic: Model, actor: Model, model: Model,
        batch: Batch, discount: float, expectile: float, num_repeat: int) -> Tuple[Model, Model, InfoDict]:

    key1, key2, key3, key4 = jax.random.split(key, 4)

    action_dim = batch.actions.shape[1]; N = batch.observations.shape[0]
    states = batch.observations
    actions = jax.random.uniform(key1, (N, action_dim), minval=-1., maxval=1.)

    next_states, rewards, terminals, _ = model(key2, states, actions)
    next_obs = next_states[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1) 
    next_a = jax.random.uniform(key3, (N * num_repeat, action_dim), minval=-1., maxval=1.)
    #rewards = jnp.ones_like(rewards)
    
    next_q1, next_q2 = target_critic(next_obs, next_a); next_value = jnp.minimum(next_q1, next_q2)
    next_value = next_value.reshape(N, num_repeat).mean(axis = 1)
    target_q = rewards + discount * (1 - terminals) * next_value

    ###### CQL ######

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, states, actions)
        critic_loss = loss(target_q, q1, expectile) + loss(target_q, q2, expectile)
        critic_loss = critic_loss.mean()

        critic_info = {
            'base_critic_loss': critic_loss,
            'base_rewards': rewards.mean(), 'base_rewards_min': rewards.min(), 'base_rewards_max': rewards.max(),  
            'base_q1': q1.mean(), 'base_q1_min': q1.min(), 'base_q1_max': q1.max(), 
            'base_q2': q2.mean(), 'base_q2_min': q2.min(), 'base_q2_max': q2.max(),
        }
        #jax.debug.print('{x} {y} {z} {w}', x=q1.mean(), y=q2.mean(), z=target_q.mean(), w=next_value.mean())
    
        return critic_loss, critic_info 

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, {**info}


# COMBO
def update_q(key: PRNGKey, critic: Model, target_critic: Model, actor: Model, model: Model,
             data_batch: Batch, model_batch: Batch, model_batch_ratio: float,
             discount: float, temperature: float, lamb: float, H: int, expectile: float, base_critic: Model, num_repeat: int) -> Tuple[Model, Model, InfoDict]:

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
        a_next = actor(s_next, 0.0).sample(seed=rng1)
        states.append(s_next)
        actions.append(a_next)
        rewards.append(rew)
        masks.append(1 - terminal)

    target_q_rollout = []
    #next_q1, next_q2 = target_critic(states[-1], actions[-1]); next_value = jnp.minimum(next_q1, next_q2)
    #next_q1, next_q2 = critic(states[-1], actions[-1]); next_value = jnp.minimum(next_q1, next_q2)
    next_value = critic(states[-1], actions[-1])
    if base_critic is not None:
        base_q1, base_q2 = base_critic(states[-1], actions[-1]); base_next_value = jnp.minimum(base_q1, base_q2)
        next_value = jnp.maximum(next_value, base_next_value)
        base_q_rollout = []
        clip_ratio_rollout = []
    value_estimate = next_value
    for i in reversed(range(H)):
        #next_q1, next_q2 = target_critic(states[i+1], actions[i+1]); next_value = jnp.minimum(next_q1, next_q2)
        #next_q1, next_q2 = critic(states[i+1], actions[i+1]); next_value = jnp.minimum(next_q1, next_q2) 
        next_value = critic(states[i+1], actions[i+1])
        if base_critic is not None:
            base_q1, base_q2 = base_critic(states[i+1], actions[i+1]); base_next_value = jnp.minimum(base_q1, base_q2)
            next_value = jnp.maximum(next_value, base_next_value)
            base_q_rollout.append(base_next_value)
            clip_ratio_rollout.append((next_value <= base_next_value).mean())
        value_estimate = rewards[i] + discount * masks[i] * (lamb * value_estimate + (1-lamb) * next_value)
        target_q_rollout.append(value_estimate)

    loss_weights = [jnp.ones_like(rewards[i])]
    for i in range(H):
        loss_weights.append(loss_weights[i] * discount * lamb * masks[i])
    target_q_rollout = list(reversed(target_q_rollout))

    if base_critic is not None:
        clip_ratio_rollout = jnp.stack(clip_ratio_rollout, axis = 0)
        base_q_rollout = jnp.stack(base_q_rollout, axis = 0) 

    #target_q_rollout = target_q_rollout[0]
    #states = states[0]
    #actions = actions[0]
    #loss_weights = loss_weights[0]

    target_q_rollout = jnp.concatenate(target_q_rollout, axis=0)
    next_states = jnp.concatenate(states[1:], axis=0)
    states = jnp.concatenate(states[:-1], axis=0)
    next_actions = jnp.concatenate(actions[1:], axis=0)
    actions = jnp.concatenate(actions[:-1], axis=0)
    rewards = jnp.concatenate(rewards, axis=0)
    loss_weights = jnp.concatenate(loss_weights[:-1], axis=0)

    #target_q_rollout = target_q_rollout.reshape(N, num_repeat)
    #target_q_rollout = jnp.take_along_axis(target_q_rollout, jnp.argmin(target_q_rollout, axis=1)[:, None], axis=1).squeeze(1)

    next_a = actor(data_batch.next_observations, temperature).sample(seed=key1)
    #next_q1, next_q2 = critic(data_batch.next_observations, next_a); next_value = jnp.minimum(next_q1, next_q2)
    next_value = critic(data_batch.next_observations, next_a)
    #next_q1, next_q2 = target_critic(data_batch.next_observations, next_a); next_value = jnp.minimum(next_q1, next_q2)
    if base_critic is not None:
        base_q1, base_q2 = base_critic(data_batch.next_observations, next_a); base_next_value = jnp.minimum(base_q1, base_q2)
        next_value = jnp.maximum(next_value, base_next_value)
        clip_ratio_data = (next_value <= base_next_value).mean()
        base_q_data = base_next_value

    #next_value = target_value(data_batch.next_observations)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_value

    ###### CQL ######

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        critic_loss_data = loss(target_q_data, q_data, 0.5) 
        bellman_error_data = jnp.where(target_q_data > q_data, 0.5, 0.5) * (q_data - target_q_data)

        q_rollout = critic.apply({'params': critic_params}, states, actions)
        critic_loss_rollout = loss(target_q_rollout, q_rollout, expectile)
        critic_loss_rollout = (critic_loss_rollout * loss_weights).reshape(-1, N, num_repeat).mean(axis=(0, 2))
        bellman_error_rollout = jnp.where(target_q_rollout > q_rollout, expectile, 1-expectile) * (q_rollout - target_q_rollout)

        critic_loss = critic_loss_data.mean() * (1 - model_batch_ratio) + critic_loss_rollout.mean() * model_batch_ratio

        q_target_data = target_critic(data_batch.observations, data_batch.actions)
        critic_reg_loss_data = (q_target_data - q_data) ** 2 
        q_target_rollout = target_critic(states, actions)
        critic_reg_loss_rollout = (q_target_rollout - q_rollout) ** 2 
        critic_reg_loss_rollout = critic_reg_loss_rollout * loss_weights
        critic_reg_loss = critic_reg_loss_data.mean() * (1 - model_batch_ratio) + critic_reg_loss_rollout.mean() * model_batch_ratio

        critic_info = {
            'critic_loss': critic_loss, 'loss_weights': loss_weights.mean(), 
            'q_data': q_data.mean(), 'q_data_min': q_data.min(), 'q_data_max': q_data.max(), 
            'q_rollout': q_rollout.mean(), 'q_rollout_min': q_rollout.min(), 'q_rollout_max': q_rollout.max(), 'q_adv': q_rollout.mean() - q_data.mean(),
            'reward_data': data_batch.rewards.mean(), 'reward_data_max': data_batch.rewards.max(), 'reward_data_min': data_batch.rewards.min(),
            'reward_model': (rewards * loss_weights).mean(), 'reward_max': (rewards * loss_weights).max(), 'reward_min': (rewards * loss_weights).min(),
            'bellman_error_data': bellman_error_data.mean(), 'bellman_error_rollout': bellman_error_rollout.mean(), 'expectile': expectile,
            'lambda': lamb.mean(), 'lambda_min': lamb.min(), 'lambda_max': lamb.max(),
            'state_max': jnp.abs(states * loss_weights[:, None]).max(), 'critic_reg_loss': critic_reg_loss,
        }
    
        if base_critic is not None:
            critic_info.update({
            'baseline_q_data': base_q_data.mean(), 'baseline_q_data_min': base_q_data.min(), 'baseline_q_data_max': base_q_data.max(),
            'baseline_q_rollout': base_q_rollout.mean(), 'baseline_q_rollout_min': base_q_rollout.min(), 'baseline_q_rollout_max': base_q_rollout.max(),
            'clip_ratio_data': jnp.mean(clip_ratio_data), 'clip_ratio_rollout': jnp.mean(clip_ratio_rollout),
            }) 
        return critic_loss + critic_reg_loss, critic_info 

    print(actions)
    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, {**info}

def update_tau_model(tau_model:Model, bellman_error: jnp.ndarray):
    def tau_model_loss_fn(tau_model_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_expectile = tau_model.apply({'params': tau_model_params}); expectile = jax.nn.sigmoid(log_expectile)
        tau_model_loss = -(log_expectile * bellman_error)
        return tau_model_loss, {}
    new_tau_model, info = tau_model.apply_gradient(tau_model_loss_fn)
    return new_tau_model, info

