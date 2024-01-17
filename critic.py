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

def _update_q(key: PRNGKey, critic: Model, target_value: Model, target_critic: Model, model: Model, actor: Model,
             batch: Batch, discount: float, lamb: float, H: float) -> Tuple[Model, InfoDict]:

    N = batch.rewards.shape[0]
    target_q_data = batch.rewards + discount * batch.masks * target_value(batch.next_observations)

    s = batch.observations
    states = [s]
    rewards, actions, masks = [], [], []
    for i in range(H):
        key, key2 = jax.random.split(key)
        a = actor(s).sample(seed=key)
        #dist, rew, mask = model(s, a, training=False); C = rew.shape[1]
        #idxs = jax.random.choice(key2, C, (N, 1, 1))
        dist, rew, mask = model(s, a, indexs=[0], training=False); C = rew.shape[1]
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
    
    num_repeat = 10;
    obs = jnp.concatenate(states[:-1], axis=0)
    next_obs = jnp.concatenate(states[1:], axis=0)
    rewards = jnp.concatenate(rewards, axis=0)
    masks = jnp.concatenate(masks, axis=0)

    N = obs.shape[0]
    obs = jnp.tile(obs, (num_repeat, 1)) 
    next_obs = jnp.tile(next_obs, (num_repeat, 1))
    rewards = jnp.tile(rewards, (num_repeat,))
    masks = jnp.tile(masks, (num_repeat,))

    a = actor(obs).sample(seed=key); next_a = actor(next_obs).sample(seed=key)
    random_action = jax.random.uniform(key, a.shape, minval=-1., maxval=1.)

    #next_v = target_value(next_obs) 
    next_v = target_critic(next_obs, next_a); 
    target_q_model = rewards + discount * masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_data, q2_data = critic.apply({'params': critic_params}, batch.observations, batch.actions)
        #critic_loss_data = ((q1_data - target_q_data)**2 + (q2_data - target_q_data)**2).mean()
        critic_loss_data = (loss(target_q_data - q1_data, 0.5) + loss(target_q_data - q2_data, 0.5)).mean()

        #q1_model, q2_model = critic.apply({'params': critic_params}, batch.observations, actions[0])
        q1_model, q2_model = critic.apply({'params': critic_params}, obs, a)
        critic_loss_model = (loss(target_q_model - q1_model, 0.5) + loss(target_q_model - q2_model, 0.5)).mean()

        obs_pi_value1, obs_pi_value2 = critic.apply({'params': critic_params}, obs, a)
        obs_pi_value1, obs_pi_value2 = obs_pi_value1.reshape((N, num_repeat)), obs_pi_value2.reshape((N, num_repeat))

        next_obs_pi_value1, next_obs_pi_value2 = critic.apply({'params': critic_params}, next_obs, next_a)
        next_obs_pi_value1, next_obs_pi_value2 = next_obs_pi_value1.reshape((N, num_repeat)), next_obs_pi_value2.reshape((N, num_repeat))

        random_value1, random_value2 = critic.apply({'params': critic_params}, obs, random_action)
        random_value1, random_value2 = random_value1.reshape((N, num_repeat)), random_value2.reshape((N, num_repeat))

        cat_q1 = jnp.concatenate([obs_pi_value1, next_obs_pi_value1, random_value1], axis=1)
        cat_q2 = jnp.concatenate([obs_pi_value2, next_obs_pi_value2, random_value2], axis=1)
        cat_q1, cat_q2 = jax.scipy.special.logsumexp(cat_q1, axis=1), jax.scipy.special.logsumexp(cat_q2, axis=1)

        conservative_loss = -(q1_data + q2_data).mean() + (cat_q1 + cat_q2).mean()
        #jax.debug.print("B {x}, {y}, {z}, {w}", x=q1_data.shape, y=target_q_data.shape, z=q1_model.shape, w=target_q_model.shape)

        #critic_loss = critic_loss_data + critic_loss_model
        critic_loss = critic_loss_data + critic_loss_model + conservative_loss * 5.0
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_data': q1_data.mean(), 'q1_model': q1_model.mean(),
            'q2_data': q2_data.mean(), 'q2_model': q2_model.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info

# COMBO
def update_q(key: PRNGKey, critic: Model, target_critic: Model, model: Model, actor: Model,
             data_batch: Batch, model_batch: Batch, discount: float, lamb: float, H: float) -> Tuple[Model, InfoDict]:

    alpha = 0.2
    mix_batch = Batch(observations=jnp.concatenate([data_batch.observations, model_batch.observations], axis=0),
                      actions=jnp.concatenate([data_batch.actions, model_batch.actions], axis=0), 
                      rewards=jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0),
                      masks=jnp.concatenate([data_batch.masks, model_batch.masks], axis=0),
                      next_observations=jnp.concatenate([data_batch.next_observations, model_batch.next_observations], axis=0),)

    next_a = actor(mix_batch.next_observations).sample(seed=key)
    next_q1, next_q2 = target_critic(mix_batch.next_observations, next_a); next_q = jnp.minimum(next_q1, next_q2)
    target_q = mix_batch.rewards + discount * mix_batch.masks * next_q

    ############

    num_repeat = 10; N = mix_batch.observations.shape[0]
    Robs = jnp.tile(mix_batch.observations, (num_repeat, 1)) 
    Rnext_obs = jnp.tile(mix_batch.next_observations, (num_repeat, 1))

    Ra = actor(Robs).sample(seed=key); Rnext_a = actor(Rnext_obs).sample(seed=key)
    Rrandom_action = jax.random.uniform(key, Ra.shape, minval=-1., maxval=1.)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, mix_batch.observations, mix_batch.actions)
        critic_loss = (loss(target_q - q1, 0.5) + loss(target_q - q2, 0.5)).mean()

        obs_pi_value1, obs_pi_value2 = critic.apply({'params': critic_params}, Robs, Ra)
        obs_pi_value1, obs_pi_value2 = obs_pi_value1.reshape((N, num_repeat)), obs_pi_value2.reshape((N, num_repeat))

        next_obs_pi_value1, next_obs_pi_value2 = critic.apply({'params': critic_params}, Rnext_obs, Rnext_a)
        next_obs_pi_value1, next_obs_pi_value2 = next_obs_pi_value1.reshape((N, num_repeat)), next_obs_pi_value2.reshape((N, num_repeat))

        random_value1, random_value2 = critic.apply({'params': critic_params}, Robs, Rrandom_action)
        random_value1, random_value2 = random_value1.reshape((N, num_repeat)), random_value2.reshape((N, num_repeat))

        cat_q1 = jnp.concatenate([obs_pi_value1, next_obs_pi_value1, random_value1], axis=1)
        cat_q2 = jnp.concatenate([obs_pi_value2, next_obs_pi_value2, random_value2], axis=1)
        cat_q1, cat_q2 = jax.scipy.special.logsumexp(cat_q1, axis=1), jax.scipy.special.logsumexp(cat_q2, axis=1)

        q1_data, q2_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        q1_model, q2_model = critic.apply({'params': critic_params}, model_batch.observations, model_batch.actions)

        conservative_loss = -(q1_data + q2_data).mean() + (cat_q1 + cat_q2).mean()
        critic_loss = critic_loss + conservative_loss * 5.0
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_data': q1_data.mean(), 'q1_model': q1_model.mean(),
            'q2_data': q2_data.mean(), 'q2_model': q2_model.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
