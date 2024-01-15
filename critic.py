from typing import Tuple

import jax.numpy as jnp
import jax

from common import Batch, InfoDict, Model, Params, PRNGKey


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


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


def _update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

def update_q(key: PRNGKey, critic: Model, target_value: Model, model: Model, actor: Model, batch: Batch,
        discount: float, lamb: float, H: float) -> Tuple[Model, InfoDict]:

    N = batch.rewards.shape[0]
    target_q_data = batch.rewards + discount * batch.masks * target_value(batch.next_observations)

    s = batch.observations
    states = [s]
    rewards, actions, masks = [], [], []
    for i in range(H):
        key, key2 = jax.random.split(key)
        a = actor(s).sample(seed=key)
        dist, rew, mask = model(s, a, training=False); C = rew.shape[1]
        idxs = jax.random.choice(key2, C, (N, 1, 1))
        mu_hats, logvar_hats = dist; s_next = mu_hats
        s_next = jnp.take_along_axis(s_next, idxs, 1).squeeze(1)
        rew = jnp.take_along_axis(rew, idxs, 1).squeeze((1, 2))
        mask = jnp.take_along_axis(mask, idxs, 1).squeeze((1, 2))
        #s_next, rew, mask = model(s, a)
        states.append(s_next)
        actions.append(a)
        rewards.append(rew)
        masks.append(mask)

    target_q_model = target_value(states[-1])
    for i in reversed(range(H)):
        #jax.debug.print("A {x}, {y}, {z}", x=rewards[i].shape, y=masks[i].shape, z=target_q_model.shape)
        target_q_model = rewards[i] + discount * masks[i] * (lamb * target_q_model + (1-lamb) * target_value(states[i])) 

    #next_v = target_value(states[1])
    #target_q_model = rewards[0] + discount * masks[0] * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_data, q2_data = critic.apply({'params': critic_params}, batch.observations, batch.actions)
        critic_loss_data = ((q1_data - target_q_data)**2 + (q2_data - target_q_data)**2).mean()

        q1_model, q2_model = critic.apply({'params': critic_params}, batch.observations, actions[0])
        critic_loss_model = (loss(target_q_model - q1_model, 0.1) + loss(target_q_model - q2_model, 0.1)).mean()
        #jax.debug.print("B {x}, {y}, {z}, {w}", x=q1_data.shape, y=target_q_data.shape, z=q1_model.shape, w=target_q_model.shape)

        critic_loss = critic_loss_data + critic_loss_model
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_data': q1_data.mean(), 'q1_model': q1_model.mean(),
            'q2_data': q2_data.mean(), 'q2_model': q2_model.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
