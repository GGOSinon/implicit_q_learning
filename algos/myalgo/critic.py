from typing import Tuple

import jax.numpy as jnp
import jax

import numpy as np
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
# COMBO
def update_q(key: PRNGKey, critic: Model, target_critic: Model, actor: Model, cql_beta: Model, model: Model,
             data_batch: Batch, model_batch: Batch, discount: float, cql_weight: float, target_beta: float) -> Tuple[Model, Model, InfoDict]:

    key1, key2, key3, key4 = jax.random.split(key, 4)

    mix_batch = Batch(observations=jnp.concatenate([data_batch.observations, model_batch.observations], axis=0),
                      actions=jnp.concatenate([data_batch.actions, model_batch.actions], axis=0), 
                      rewards=jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0),
                      masks=jnp.concatenate([data_batch.masks, model_batch.masks], axis=0),
                      next_observations=jnp.concatenate([data_batch.next_observations, model_batch.next_observations], axis=0),)

    next_obs, _, _, _ = model(key1, data_batch.observations, data_batch.actions)
    #next_a_data = actor(data_batch.next_observations).sample(seed=key1)
    next_a_data = actor(next_obs).sample(seed=key1)
    next_q1_data, next_q2_data = target_critic(data_batch.next_observations, next_a_data); next_q_data = jnp.minimum(next_q1_data, next_q2_data)

    next_a_model = actor(model_batch.next_observations).sample(seed=key1)
    next_q1_model, next_q2_model = target_critic(model_batch.next_observations, next_a_model); next_q_model = jnp.minimum(next_q1_model, next_q2_model)

    target_q_data = data_batch.rewards + discount * data_batch.masks * next_q_data
    target_q_model = model_batch.rewards + discount * model_batch.masks * next_q_model

    ###### CQL ######

    num_repeat = 10; N = mix_batch.observations.shape[0]
    Robs = mix_batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
    Rnext_obs = mix_batch.next_observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)

    Ra_dist = actor(Robs); Rnext_a_dist = actor(Rnext_obs)
    Ra = Ra_dist.sample(seed=key2); Rnext_a = Rnext_a_dist.sample(seed=key3)
    log_prob_Ra = Ra_dist.log_prob(Ra); log_prob_Rnext_a = Rnext_a_dist.log_prob(Rnext_a)
    Rrandom_action = jax.random.uniform(key4, Ra.shape, minval=-1., maxval=1.)
    log_prob_rand = np.log(0.5) * Ra.shape[-1]

    if cql_beta is not None:
        log_beta = cql_beta(); beta = jnp.exp(log_beta)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_data, q2_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        q1_model, q2_model = critic.apply({'params': critic_params}, model_batch.observations, model_batch.actions)

        #q1, q2 = critic.apply({'params': critic_params}, mix_batch.observations, mix_batch.actions)
        #critic_loss = (loss(target_q - q1, 0.5) + loss(target_q - q2, 0.5)).mean()
        critic_loss_data = loss(target_q_data - q1_data, 0.9) + loss(target_q_data - q2_data, 0.9)
        critic_loss_model = loss(target_q_model - q1_model, 0.5) + loss(target_q_model - q2_model, 0.5)
        critic_loss = jnp.concatenate([critic_loss_data, critic_loss_model], axis=0)
        critic_loss = critic_loss_data
        critic_loss = critic_loss.mean()

        obs_pi_value1, obs_pi_value2 = critic.apply({'params': critic_params}, Robs, Ra)
        obs_pi_value1, obs_pi_value2 = obs_pi_value1 - log_prob_Ra, obs_pi_value2 - log_prob_Ra 
        obs_pi_value1, obs_pi_value2 = obs_pi_value1.reshape((N, num_repeat)), obs_pi_value2.reshape((N, num_repeat))

        next_obs_pi_value1, next_obs_pi_value2 = critic.apply({'params': critic_params}, Robs, Rnext_a)
        next_obs_pi_value1, next_obs_pi_value2 = next_obs_pi_value1 - log_prob_Rnext_a, next_obs_pi_value2 - log_prob_Rnext_a 
        next_obs_pi_value1, next_obs_pi_value2 = next_obs_pi_value1.reshape((N, num_repeat)), next_obs_pi_value2.reshape((N, num_repeat))

        random_value1, random_value2 = critic.apply({'params': critic_params}, Robs, Rrandom_action)
        random_value1, random_value2 = random_value1 - log_prob_rand, random_value2 - log_prob_rand
        random_value1, random_value2 = random_value1.reshape((N, num_repeat)), random_value2.reshape((N, num_repeat))

        cat_q1 = jnp.concatenate([obs_pi_value1, next_obs_pi_value1, random_value1], axis=1)
        cat_q2 = jnp.concatenate([obs_pi_value2, next_obs_pi_value2, random_value2], axis=1)
        cat_q1, cat_q2 = jax.scipy.special.logsumexp(cat_q1, axis=1), jax.scipy.special.logsumexp(cat_q2, axis=1)

        conservative_loss = -(q1_data + q2_data).mean() + (cat_q1 + cat_q2).mean()

        if cql_beta is None:
            critic_loss = critic_loss + conservative_loss * cql_weight
        else:
            critic_loss = critic_loss + (conservative_loss - target_beta) * cql_weight * beta
        return critic_loss, {
            'critic_loss': critic_loss, 'conservative_loss': conservative_loss,
            'q1_data': q1_data.mean(), 'q1_model': q1_model.mean(),
            'q2_data': q2_data.mean(), 'q2_model': q2_model.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    if cql_beta is None:
        new_beta, beta_info = None, {}
    else:
        conservative_loss = info['conservative_loss']
        def cql_beta_loss_fn(cql_beta_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
            log_beta = cql_beta.apply({'params': cql_beta_params}); beta = jnp.exp(log_beta)
            cql_beta_loss = -beta * cql_weight * (conservative_loss - target_beta)
            return cql_beta_loss, {
                'cql_beta_loss': cql_beta_loss,
                'beta': beta,
            }

        new_beta, beta_info = cql_beta.apply_gradient(cql_beta_loss_fn)

    return new_critic, new_beta, {**info, **beta_info}
