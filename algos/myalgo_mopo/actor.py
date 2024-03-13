from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss

def update_actor(key: PRNGKey, actor: Model, critic: Model, model: Model,
        batch: Batch, discount: float, temperature: float, sac_alpha: float) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations, training=True, rngs={'dropout': key})
        a = dist.sample(seed=key); log_probs = dist.log_prob(a)
        q1, q2 = critic(batch.observations, a)
        q = jnp.minimum(q1, q2)

        actor_loss = -q.mean() + sac_alpha * log_probs.mean()

        policy_std = dist.scale.diag if hasattr(dist, 'scale') else dist.distribution.scale.diag
        return actor_loss, {'actor_loss': actor_loss, 'policy_std': policy_std.mean(), 'log_probs': log_probs.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def reinforce_update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model, model: Model,
        batch: Batch, discount: float, temperature: float, sac_alpha: float) -> Tuple[Model, InfoDict]:

    v = value(batch.observations)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations, training=True, rngs={'dropout': key})
        a = dist.sample(seed=key); log_probs = dist.log_prob(a)
        q1, q2 = critic(batch.observations, a)
        q = jnp.minimum(q1, q2)
        adv = jax.lax.stop_gradient(q - v)

        actor_loss = -(adv * log_probs).mean() + sac_alpha * log_probs.mean()

        policy_std = dist.scale.diag if hasattr(dist, 'scale') else dist.distribution.scale.diag
        return actor_loss, {'actor_loss': actor_loss, 'policy_std': policy_std.mean(), 'log_probs': log_probs.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def awr_update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model, model: Model,
        batch: Batch, discount: float, temperature: float, sac_alpha: float) -> Tuple[Model, InfoDict]:

    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations, training=True, rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)

        actor_loss = -(exp_a * log_probs).mean() #+ sac_alpha * log_probs.mean()

        policy_std = dist.scale.diag if hasattr(dist, 'scale') else dist.distribution.scale.diag
        return actor_loss, {'actor_loss': actor_loss, 'policy_std': policy_std.mean(), 'log_probs': log_probs.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def gae_update_actor(key: PRNGKey, actor: Model, critic: Model, model: Model,
        batch: Batch,
        discount: float, temperature: float, sac_alpha: float, lamb: float, H: int, expectile: float) -> Tuple[Model, InfoDict]:

    num_repeat = 1; N = batch.observations.shape[0]
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        Robs = batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
        dist = actor.apply({'params': actor_params}, Robs, temperature, training=True, rngs={'dropout': key})
        Ra = dist.sample(seed=key) 
        states, rewards, actions, masks, weights = [Robs], [], [Ra], [], [jnp.ones(N*num_repeat)]
        keys = [key]

        q_values = []
        q1, q2 = critic(Robs, Ra); q_values.append(jnp.minimum(q1, q2))
        for i in range(H):
            s, a = states[-1], actions[-1]
            rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4); keys.append(key0)
            s_next, rew, terminal, _ = model(rng1, s, a)
            dist = actor.apply({'params': actor_params}, s_next, temperature, training=True)
            a_next = dist.sample(seed=rng2) 
            q1, q2 = critic(s_next, a_next); q_values.append(jnp.minimum(q1, q2))
            states.append(s_next)
            actions.append(a_next)
            rewards.append(rew)
            masks.append(1 - terminal)
            weights.append(weights[-1] * masks[-1] * discount)

        q_rollout, loss_weights = [], []
        q1, q2 = critic(states[-1], actions[-1])
        value_estimate = jnp.minimum(q1, q2)
        for i in reversed(range(H)):
            q1, q2 = critic(states[i+1], actions[i+1])
            value_estimate = rewards[i] + discount * masks[i] * (lamb * value_estimate + (1-lamb) * jnp.minimum(q1, q2))
            #q_rollout.append(lamb * value_estimate + (1 - lamb) * q_values[i])
            q_rollout.append(value_estimate)
            loss_weights.append(jnp.where(value_estimate > q_values[i], expectile, 1-expectile)) 
        q_rollout = list(reversed(q_rollout))
        loss_weights = list(reversed(loss_weights))

        dists, log_probs = [], []
        for i in range(H+1):
            dist = actor.apply({'params': actor_params}, states[i], training=True)
            action = dist.sample(seed=keys[i])
            dists.append(dist)
            log_probs.append(dist.log_prob(action))

        policy_std = [dist.scale.diag if hasattr(dist, 'scale') else dist.distribution.scale.diag for dist in dists]

        q_values = jnp.stack(q_values[:-1], axis = 1)
        loss_weights = jnp.stack(loss_weights, axis=1) # [N, H]
        q_rollout = jnp.stack(q_rollout, axis=1) # [N, H] 
        log_probs = jnp.stack(log_probs[:-1], axis=1) # [N, H]
        policy_std = jnp.stack(policy_std[:-1], axis=1)
        weights = jnp.stack(weights[:-1], axis=1) # [N, H]

        actor_loss = -(lamb * loss_weights * q_rollout + (1-lamb) * q_values) + sac_alpha * log_probs
        actor_loss = (actor_loss * weights).mean()
        #actor_loss = -(q_rollout * weights).mean() + sac_alpha * log_probs.mean()
        #jax.debug.print('{x}, {y}', x=actor_loss, y=log_probs.mean())

        return actor_loss, {'actor_loss': actor_loss, 'q_rollout': q_rollout, 'policy_std': policy_std.mean(), 'log_probs': log_probs.mean(), 'adv_weights': (loss_weights * weights).mean() / weights.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info

def update_all(key: PRNGKey, actor: Model, critic: Model, target_critic: Model, model: Model, sac_alpha: Model, base_critic: Model,
               data_batch: Batch, model_batch: Batch, model_batch_ratio: float,
               discount: float, lamb: float, H: int, expectile: float, target_entropy: float,) -> Tuple[Model, InfoDict]:

    mix_batch = Batch(observations=jnp.concatenate([data_batch.observations, model_batch.observations], axis=0),
                      actions=jnp.concatenate([data_batch.actions, model_batch.actions], axis=0),
                      rewards=jnp.concatenate([data_batch.rewards, model_batch.rewards], axis=0),
                      masks=jnp.concatenate([data_batch.masks, model_batch.masks], axis=0),
                      next_observations=jnp.concatenate([data_batch.next_observations, model_batch.next_observations], axis=0),
                      returns_to_go=None)

    # Actor update
    log_alpha = sac_alpha(); alpha = jnp.exp(log_alpha)
    
    num_repeat = 10; N = model_batch.observations.shape[0]
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        Robs = model_batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
        dist = actor.apply({'params': actor_params}, Robs, training=True)
        Ra = dist.sample(seed=key); log_prob = dist.log_prob(Ra)
        states, rewards, actions, masks, log_probs, dists, weights = [Robs], [], [Ra], [], [log_prob], [dist], [jnp.ones(N*num_repeat)]

        q_values = []
        q1, q2 = critic(Robs, Ra); q_values.append(jnp.minimum(q1, q2))
        keys = [key]
        for i in range(H):
            s, a = states[-1], actions[-1]
            rng1, rng2, rng3, key0 = jax.random.split(keys[-1], 4); keys.append(key0)
            s_next, rew, terminal, _ = model(rng1, s, a)
            dist = actor.apply({'params': actor_params}, s_next, training=True)
            a_next = dist.sample(seed=rng2); log_prob = dist.log_prob(a_next)
            q1, q2 = critic(s_next, a_next); q_values.append(jnp.minimum(q1, q2))
            states.append(s_next)
            actions.append(a_next)
            rewards.append(rew)
            masks.append(1 - terminal)
            log_probs.append(log_prob)
            dists.append(dist)
            weights.append(weights[-1] * masks[-1] * discount)

        policy_std = [dist.scale.diag if hasattr(dist, 'scale') else dist.distribution.scale.diag for dist in dists]

        q_rollout, loss_weights = [], []
        q1, q2 = target_critic(states[-1], actions[-1])
        value_estimate = jnp.minimum(q1, q2)
        for i in reversed(range(H)):
            value_estimate = rewards[i] + discount * masks[i] * (lamb * value_estimate + (1-lamb) * q_values[i+1])
            q_rollout.append(value_estimate)
            loss_weights.append(jnp.where(value_estimate > q_values[i], expectile, 1-expectile))
        q_rollout = list(reversed(q_rollout))
        loss_weights = list(reversed(loss_weights))

        loss_weights = jnp.concatenate(loss_weights, axis=0) # [N*R*H]
        q_rollout = jnp.concatenate(q_rollout, axis=0) # [N*R*H] 
        log_probs = jnp.concatenate(log_probs[:-1], axis=0) # [N*R*H]
        policy_std = jnp.concatenate(policy_std[:-1], axis=0)
        weights = jnp.concatenate(weights[:-1], axis=0) # [N*R*H]
        actor_loss = -(q_rollout * weights * loss_weights).mean() + alpha * log_probs.mean()
        #jax.debug.print('{x}, {y}', x=actor_loss, y=log_probs.mean())

        trajs = {'states': jnp.concatenate(states[:-1], axis=0),
                 'actions': jnp.concatenate(actions[:-1], axis=0),
                 'rewards': jnp.concatenate(rewards, axis=0),
                 'target_q': q_rollout,
                 'log_probs': log_probs,
                }

        return actor_loss, {'actor_loss': actor_loss, 'q_rollout': q_rollout, 'policy_std': policy_std.mean(), 'log_probs': log_probs.mean(), 'traj': trajs}

    new_actor, actor_info = actor.apply_gradient(actor_loss_fn)

    ### Critic update
    traj = actor_info.pop('traj')
    states, actions, target_q_rollout = traj['states'], traj['actions'], traj['target_q']

    if base_critic is not None:
        base_q1, base_q2 = base_critic(states, actions); base_q_rollout = jnp.minimum(base_q1, base_q2)
        target_q_rollout = jnp.maximum(target_q_rollout, base_q_rollout)
        clip_ratio_rollout = (target_q_rollout <= base_q_rollout).mean()

    next_a = actor(data_batch.next_observations).sample(seed=key)
    next_q1, next_q2 = target_critic(data_batch.next_observations, next_a); next_q = jnp.minimum(next_q1, next_q2)
    target_q_data = data_batch.rewards + discount * data_batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_data, q2_data = critic.apply({'params': critic_params}, data_batch.observations, data_batch.actions)
        critic_loss_data = (loss(target_q_data, q1_data, 0.5) + loss(target_q_data, q2_data, 0.5))

        q1_rollout, q2_rollout = critic.apply({'params': critic_params}, states, actions)
        critic_loss_rollout = (loss(target_q_rollout, q1_rollout, expectile) + loss(target_q_rollout, q2_rollout, expectile))
        critic_loss_rollout = critic_loss_rollout.reshape(H, N, num_repeat).mean((0, 2))
        
        critic_loss = critic_loss_data.mean() * (1 - model_batch_ratio) + critic_loss_rollout.mean() * model_batch_ratio
        #critic_loss = jnp.concatenate([critic_loss_data, critic_loss_rollout], axis=0)
        #critic_loss = critic_loss.mean()
        critic_info = {
            'critic_loss': critic_loss,
            'q1_data': q1_data.mean(), 'q1_rollout': q1_rollout.mean(), 'q1_adv': q1_rollout.mean() - q1_data.mean(),
            'q2_data': q2_data.mean(), 'q2_rollout': q2_rollout.mean(), 'q2_adv': q2_rollout.mean() - q2_data.mean(),
            'reward_data': data_batch.rewards.mean(), 'reward_data_max': data_batch.rewards.max(),
            'reward_model': traj['rewards'].mean(), 'reward_max': traj['rewards'].max(),
        }

        if base_critic is not None:
            critic_info.update({
                'baseline_q_rollout': base_q_rollout.mean(), 'baseline_q_rollout_min': base_q_rollout.min(), 'baseline_q_rollout_max': base_q_rollout.max(), 'clip_ratio_rollout': clip_ratio_rollout,
            })
        return critic_loss, critic_info 

    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)

    #### SAC autotune
    log_probs = traj['log_probs'].mean() + target_entropy

    def alpha_loss_fn(alpha_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_alpha = sac_alpha.apply({'params': alpha_params})
        alpha_loss = -(log_alpha * log_probs).mean()

        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': jnp.exp(log_alpha), 'log_probs': log_probs.mean()}

    new_alpha, alpha_info = sac_alpha.apply_gradient(alpha_loss_fn) 

    return new_actor, new_critic, new_alpha, actor_info, critic_info, alpha_info

def update_alpha(key: PRNGKey, actor: Model, sac_alpha: Model,
        batch: Batch, target_entropy: float) -> Tuple[Model, InfoDict]:

    dist = actor(batch.observations); a = dist.sample(seed=key)
    log_probs = dist.log_prob(a) + target_entropy

    def alpha_loss_fn(alpha_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_alpha = sac_alpha.apply({'params': alpha_params})
        alpha_loss = -(log_alpha * log_probs).mean()

        if hasattr(dist, 'scale'):
            policy_std = dist.scale.diag
        else:
            policy_std = dist.distribution.scale.diag

        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': jnp.exp(log_alpha), 'policy_std': policy_std.mean(), 'log_probs': log_probs.mean()}

    new_alpha, info = sac_alpha.apply_gradient(alpha_loss_fn)

    return new_alpha, info
