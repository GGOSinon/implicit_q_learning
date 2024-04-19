from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey
from common import expectile_loss as loss

def gae_update_actor(key: PRNGKey, actor: Model, critic: Model, model: Model,
        batch: Batch,
        discount: float, temperature: float, sac_alpha: float, H: int, expectile: float, num_repeat: int) -> Tuple[Model, InfoDict]:

    N = batch.observations.shape[0]
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        Robs = batch.observations[:, None, :].repeat(repeats=num_repeat, axis=1).reshape(N * num_repeat, -1)
        dist = actor.apply({'params': actor_params}, Robs, temperature, training=True, rngs={'dropout': key})
        Ra = dist.sample(seed=key) 
        states, rewards, actions, masks, weights = [Robs], [], [Ra], [], [jnp.ones(N*num_repeat)]
        keys = [key]

        q_values = []
        q_values.append(critic(Robs, Ra))
        for i in range(H):
            s, a = states[i], actions[i]
            rng1, rng2, rng3, key0 = jax.random.split(keys[i], 4); keys.append(key0)
            s_next, rew, terminal, _ = model(rng1, s, a)
            dist = actor.apply({'params': actor_params}, s_next, temperature, training=True)
            a_next = dist.sample(seed=rng2) 
            q_values.append(critic(s_next, a_next)) 
            states.append(s_next)
            actions.append(a_next)
            rewards.append(rew)
            masks.append(1 - terminal)
            weights.append(weights[-1] * masks[-1] * discount)

        dists, log_probs = [], []
        for i in range(H+1):
            dist = actor.apply({'params': actor_params}, states[i], training=True)
            action = dist.sample(seed=keys[i])
            dists.append(dist)
            log_probs.append(dist.log_prob(action))

        policy_std = [dist.scale.diag if hasattr(dist, 'scale') else dist.distribution.scale.diag for dist in dists]

        actions = jnp.stack(actions, axis=1)
        q_values = jnp.stack(q_values[:-1], axis = 1)
        log_probs = jnp.stack(log_probs[:-1], axis=1) # [N, H]
        policy_std = jnp.stack(policy_std[:-1], axis=1)
        weights = jnp.stack(weights[:-1], axis=1) # [N, H]

        actor_loss = -q_values + sac_alpha * log_probs
        actor_loss = (actor_loss * weights).mean()
        return actor_loss, {'actor_loss': actor_loss, 
                            'policy_std': (policy_std * weights[:, :, None]).mean() / weights.mean(),
                            'log_probs': (log_probs * weights).mean() / weights.mean(),
                            'abs_actions': jnp.abs(actions).mean()
                            }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info

def update_alpha(key: PRNGKey, log_probs: jnp.ndarray, sac_alpha: Model, target_entropy: float) -> Tuple[Model, InfoDict]:
    log_probs = log_probs + target_entropy
    def alpha_loss_fn(alpha_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        log_alpha = sac_alpha.apply({'params': alpha_params})
        alpha_loss = -(log_alpha * log_probs).mean()
        return alpha_loss, {'alpha_loss': alpha_loss, 'alpha': jnp.exp(log_alpha)}

    new_alpha, info = sac_alpha.apply_gradient(alpha_loss_fn)

    return new_alpha, info
