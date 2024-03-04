"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

import dreamerv3
import dreamerv3.embodied
import dreamerv3.jaxutils as jaxutils
#import dreamerv3.ninjax as nj
from dreamerv3.agent import WorldModel as DreamerV3WorldModel
from .dreamer_model import WorldModel
from dreamerv3.embodied.envs import from_gym

import gym

sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
from common import Batch, InfoDict, Model, PRNGKey, MLP, Params
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def get_dist(state):
    logit = state['logit'].astype(jnp.float32)
    return tfd.Independent(jaxutils.OneHotDist(logit), 1)

def dyn_loss(post, prior, impl='kl', free=1.0):
    loss = get_dist(sg(post)).kl_divergence(get_dist(prior))
    loss = jnp.maximum(loss, free)
    return loss

def rep_loss(post, prior, impl='kl', free=1.0):
    loss = get_dist(post).kl_divergence(get_dist(sg(prior)))
    loss = jnp.maximum(loss, free)
    return loss

@jax.jit
def _update_jit(
    rng: PRNGKey, model: Model, batch: Batch 
) -> Tuple[PRNGKey, Model, InfoDict]:

    N, T = batch['vector'].shape[:2]
    s = batch['vector']
    a = batch['action']
    r = batch['reward']
    mask = 1.0 - batch['is_terminal']; done = batch['is_terminal']
    #s, a, r, sN, mask = batch.observations, batch.actions, batch.rewards, batch.next_observations, batch.masks
    # s: [N, T, d_obs]
    # a: [N, T, d_act]
    # r: [N, T]
    # sN: [N, T]
    # mask: [N, T]
    
    is_first = jnp.concatenate([jnp.ones((N, 1)), done[:, :-1]], axis=1)
    #jax.debug.print("r_hat:{r}, s_hat:{s}, 1-d:{x}", x=mask.mean(), r=r_hat.mean(), s=s_hat.mean())

    def loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        #sN_hat, r_hat, done_hat, is_first
        _, _, _, info = model.apply({'params': params}, rng, s, a, is_first, training=True)
        # h' = f(h, enc(h, s), a)
        # post: dyn(h')
        # prior: enc(h', s')
        # rew: rew(h', enc(h, s))
        # cont: cont(h', enc(h, s))
        # sN_hat: dec(post)

        post, prior, r_hat, cont_hat, s_hat = info['post'], info['prior'], info['reward'], info['cont'], info['s_hat']
        
        loss_dyn = dyn_loss(post, prior)
        loss_rep = rep_loss(post, prior)
        
        print(r.shape)
        loss_rew = -r_hat.log_prob(r)
        loss_cont = -cont_hat.log_prob(mask)
        loss_rec = -s_hat.log_prob(s)

        print(loss_rew.shape, loss_cont.shape, loss_rec.shape, loss_dyn.shape, loss_rep.shape)

        loss_pred = loss_rew + loss_cont + loss_rec

        loss = (1.0 * loss_pred + 0.5 * loss_dyn + 0.1 * loss_rep).sum()
        return loss, {
            'lossR': loss_rew.sum(),
            'lossD': loss_dyn.sum(),
            'lossP': loss_rep.sum() + loss_rec.sum(),
            'lossM': loss_cont.sum(),
        }

    new_model, info = model.apply_gradient(loss_fn)

    return rng, new_model, info

class Learner(object):
    def __init__(self,
                 seed: int,
                 env_name: str,
                 batch_size: int,
                 lr: float = 3e-4,
                 model_hidden_dims: Sequence[int] = None,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "cosine",
                 T: int = 16,
                 **kwargs):

        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        env = gym.make(env_name)
        #print(dropout_rate, model_hidden_dims, max_steps)
        observations = env.observation_space.sample()[None, None, :].repeat(T, axis=1)
        actions = env.action_space.sample()[None, None, :].repeat(T, axis=1)

        rng = jax.random.PRNGKey(seed)
        rng, model_key = jax.random.split(rng, 2)

        self.obs_dim = obs_dim = observations.shape[-1]
        self.action_dim = action_dim = actions.shape[-1]

        #model_def = WorldModel(model_hidden_dims, obs_dim, action_dim, dropout_rate=dropout_rate)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=lr)

        #model = Model.create(model_def,
        #                     inputs=[model_key, observations, actions],
        #                     tx=optimiser)

        if True:
            model_def = WorldModel(model_hidden_dims, obs_dim, action_dim, training=True, dropout_rate=dropout_rate)
            model = Model.create(model_def,
                                 inputs=[model_key, rng, observations, actions, np.zeros_like(actions)[:, :, 0]],
                                 tx=optimiser)
        else:
            filename = path.Path('~/logdir/run1/checkpoint.ckpt')
            ckpt = basics.unpack(filename.read('rb'))
            parsed, other = dreamerv3.embodied.Flags(configs=['defaults']).parse_known(None)
            print(parsed)
            print(other)
            config = dreamerv3.embodied.Config(dreamerv3.agent.Agent.configs['defaults'])
            #for name in parsed.configs:
            #    config = config.update(dreamerv3.Agent.configs[name])
            #config = dreamerv3.embodied.Flags(config)
            print(config)

            config = dreamerv3.embodied.Config(dreamerv3.configs['defaults'])
            config = config.update(dreamerv3.configs['medium'])
            config = config.update({
                'logdir': f'./logdir/{env_name}',
                'run.log_every': 30,
                'batch_size': 16,
                'jax.prealloc': False,
                'encoder.mlp_keys': 'vector',
                'decoder.mlp_keys': 'vector',
                'encoder.cnn_keys': '$^',
                'decoder.cnn_keys': '$^',
            })
            #config = dreamerv3.embodied.Flags(config).parse()
            env = from_gym.FromGym(env, obs_key='vector')
            env = dreamerv3.wrap_env(env, config)
            env = dreamerv3.embodied.BatchEnv([env], parallel=False)
            model = DreamerV3WorldModel(env.obs_space, env.act_space, config, name='wm')

        self.model = model
        self.rng = rng
        #self.train_fn = jax.jit(nj.pure(model.train))
        #self.init_fn = jax.jit(nj.pure(lambda: model.initial(self.batch_size)))
        self.model_state = {}
        self.batch_size = batch_size

    def preprocess(self, batch: Batch):
        return {'vector': batch.observations,
                'action': batch.actions,
                'reward': batch.rewards,
                'is_first': jnp.zeros_like(batch.masks),
                'is_last': 1.0 - batch.masks,
                'is_terminal': 1.0 - batch.masks,
                'cont': batch.masks}

    def update(self, batch: Batch) -> InfoDict:
        batch = self.preprocess(batch)
        rng = jax.random.PRNGKey(42)
        '''
        state = self.init_fn(self.model_state, rng)[0]

        results, self.model_state = self.train_fn(self.model_state, rng, batch, state)
        next_state, outs, info = results
        '''

        rng, new_model, info = _update_jit(rng, self.model, batch)
        self.model = new_model

        return info

    def evaluate(self, batch: Batch) -> InfoDict:
        batch = self.preprocess(batch)
        rng = jax.random.PRNGKey(42)

        rng, new_model, info = _update_jit(rng, self.model, batch)

        return info
