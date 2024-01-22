from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    adv = q - v
    exp_a = jnp.exp(adv * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        #actor_loss = -(log_probs).mean()
        actor_loss = -(exp_a * log_probs).mean()
        #actor_loss = -(adv * log_probs).mean()
        jax.debug.print('{x}, {y}', x=adv.max(), y=log_probs.min())

        return actor_loss, {'actor_loss': actor_loss, 'adv': adv}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        a = dist.sample(seed=key)

        q1, q2 = critic(batch.observations, a)
        q = jnp.minimum(q1, q2)
        log_probs = dist.log_prob(a)
        actor_loss = -q.mean() + 0.2 * log_probs.mean()
        #jax.debug.print('{x}, {y}', x=actor_loss, y=log_probs.mean())

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
