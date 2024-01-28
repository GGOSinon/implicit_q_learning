from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey

def update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model, model: Model,
        batch: Batch, discount: float, temperature: float, sac_alpha: float) -> Tuple[Model, InfoDict]:

    v = value(batch.observations)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations, training=True, rngs={'dropout': key})
        a = dist.sample(seed=key); log_probs = dist.log_prob(a)
        q1, q2 = critic(batch.observations, a)
        q = jnp.minimum(q1, q2)

        actor_loss = -q.mean() + sac_alpha * log_probs.mean()
        #jax.debug.print('{x}, {y}', x=actor_loss, y=log_probs.mean())

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def gae_update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model, model: Model,
        batch: Batch, discount: float, temperature: float, sac_alpha: float, lamb=0.95, H=5) -> Tuple[Model, InfoDict]:

    rng1, rng2, key = jax.random.split(key, 3)
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        states, rewards, actions, masks, log_probs = [batch.observations], [], [], [], []
        dist = actor.apply({'params': actor_params}, s, training=True, rngs={'dropout': key})
        a = dist.sample(seed=rng1); log_prob = dist.log_prob(a)
        for i in range(H):
            s = states[-1]
            s_next, rew, terminal, _ = model(rng2, s, a)
            dist = actor.apply({'params': actor_params}, s_next, training=True, rngs={'dropout': key})
            a = dist.sample(seed=rng1); log_prob = dist.log_prob(a)
            states.append(s_next)
            actions.append(a_next)
            rewards.append(rew)
            masks.append(1 - terminal)
            log_probs.append(log_prob)

        dist = actor.apply({'params': actor_params}, states[-1], training=True, rngs={'dropout': key})
        a = dist.sample(seed=key)
        actions.append(a)

        q_rollout = []
        q1, q2 = critic(states[-1], actions[-1])
        value_estimate = jnp.minimum(q1, q2)
        for i in reversed(range(H)):
            q1, q2 = critic(states[i+1], actions[i+1])
            value_estimate = rewards[i] + discount * masks[i] * (lamb * value_estimate + (1-lamb) * jnp.minimum(q1, q2))
            q_rollout.append(value_estimate)

        q_rollout = jnp.stack(q_rollout, axis=1) # [N, H] 
        log_probs = jnp.stack(log_probs, axis=1)
        actor_loss = -q_rollout.sum(axis=1).mean() + sac_alpha * log_probs.sum(axis=1).mean()
        #jax.debug.print('{x}, {y}', x=actor_loss, y=log_probs.mean())

        return actor_loss, {'actor_loss': actor_loss, 'q_rollout': q_rollout}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info

def update_alpha(key: PRNGKey, actor: Model, sac_alpha: Model,
        batch: Batch, target_entropy: float) -> Tuple[Model, InfoDict]:

    dist = actor(batch.observations); a = dist.sample(seed=key)
    log_probs = dist.log_prob(a) + target_entropy

    def alpha_loss_fn(alpha_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_alpha = sac_alpha.apply({'params': alpha_params})
        alpha_loss = -(log_alpha * log_probs).mean()

        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': jnp.exp(log_alpha), 'policy_std': dist.distribution.scale.diag.mean(), 'log_probs': log_probs.mean()}

    new_alpha, info = sac_alpha.apply_gradient(alpha_loss_fn)

    return new_alpha, info
