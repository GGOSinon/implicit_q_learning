from typing import Tuple

import jax.numpy as jnp
import jax

import numpy as np
from common import Batch, InfoDict, Model, Params, PRNGKey


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2) # / (2 * expectile * (1 - expectile))


def update_v(critic: Model, value: Model, batch: Batch,
             expectile: float) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info
# COMBO
def update_q(key: PRNGKey, critic: Model, target_critic: Model, value: Model, actor: Model, cql_beta: Model, model: Model,
             data_batch: Batch, model_batch: Batch, discount: float, cql_weight: float, target_beta: float, max_q_backup: bool,
             lamb: float, H: int) -> Tuple[Model, Model, InfoDict]:

    key1, key2, key3, key4 = jax.random.split(key, 4)
 
    num_repeat = 50; N = model_batch.observations.shape[0]
    Robs = model_batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    Ra = model_batch.actions[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    #Ra = actor(Robs).sample(seed=key1)
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

    q1, q2 = target_critic(states[-1], actions[-1])
    target_q_rollout = []
    value_estimate = jnp.minimum(q1, q2)
    for i in reversed(range(H)):
        q1, q2 = target_critic(states[i+1], actions[i+1])
        value_estimate = rewards[i] + discount * masks[i] * (lamb * value_estimate + (1-lamb) * jnp.minimum(q1, q2))
        target_q_rollout.append(value_estimate)
    target_q_rollout = list(reversed(target_q_rollout))

    target_q_rollout = jnp.concatenate(target_q_rollout, axis=0)
    states = jnp.concatenate(states[:-1], axis=0)
    actions = jnp.concatenate(actions[:-1], axis=0)

    #target_q_rollout = target_q_rollout.reshape(N, num_repeat)
    #target_q_rollout = jnp.take_along_axis(target_q_rollout, jnp.argmin(target_q_rollout, axis=1)[:, None], axis=1).squeeze(1)

    next_a = actor(data_batch.next_observations).sample(seed=key1)
    next_q1, next_q2 = target_critic(data_batch.next_observations, next_a); next_q = jnp.minimum(next_q1, next_q2)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_q

    #target_q_data = jnp.maximum(target_q_data, 0.)
    #target_q_rollout = jnp.maximum(target_q_rollout, 0.)

    #rollout_ratio = (target_q_rollout[-1] > target_q_td).mean()
    #target_q = jnp.maximum(target_q_rollout, target_q_td)
    #target_q = target_q_rollout

    ###### CQL ######

    if max_q_backup:
        next_q1, next_q2 = target_critic(Rnext_obs, Rnext_a)
        next_q1, next_q2 = next_q1.reshape(N, num_repeat), next_q2.reshape(N, num_repeat)
        next_q = jnp.minimum(next_q1, next_q2)
        next_q = jnp.take_along_axis(next_q, jnp.argmax(next_q, axis=1)[:, None], axis=1).squeeze(1)

        target_q = batch.rewards + discount * batch.masks * next_q

    if cql_beta is not None:
        log_beta = cql_beta(); beta = jnp.exp(log_beta)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_data, q2_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        critic_loss_data = (loss(target_q_data - q1_data, 0.5) + loss(target_q_data - q2_data, 0.5)).mean()

        q1_rollout, q2_rollout = critic.apply({'params': critic_params}, states, actions)
        critic_loss_rollout = (loss(target_q_rollout - q1_rollout, 0.1) + loss(target_q_rollout - q2_rollout, 0.1)).mean()

        critic_loss = critic_loss_data + critic_loss_rollout
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_data': q1_data.mean(), 'q1_rollout': q1_rollout.mean(), 'q1_rollout_min': q1_rollout.min(), 'q1_adv': q1_rollout.mean() - q1_data.mean(),
            'q2_data': q2_data.mean(), 'q2_rollout': q2_rollout.mean(), 'q1_rollout_min': q1_rollout.min(),  'q2_adv': q2_rollout.mean() - q2_data.mean(),
            'reward_data': data_batch.rewards.mean(), 'reward_data_max': data_batch.rewards.max(), 'reward_data_min': data_batch.rewards.min(),
            'reward_model': jnp.concatenate(rewards, axis=0).mean(), 'reward_max': jnp.concatenate(rewards, axis=0).max(),'reward_min': jnp.concatenate(rewards, axis=0).max(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, {**info}
