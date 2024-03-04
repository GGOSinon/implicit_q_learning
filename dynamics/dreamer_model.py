"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from jax.tree_util import tree_map
from tensorflow_probability.substrates import jax as tfp 
tfd = tfp.distributions

import dreamerv3
import dreamerv3.embodied
import dreamerv3.jaxutils as jaxutils
#import dreamerv3.ninjax as nj
from dreamerv3.agent import WorldModel as DreamerV3WorldModel
from dreamerv3.embodied.envs import from_gym

sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = lambda x: jax.tree_util.tree_map(lambda _x: _x.astype(jnp.float32), x)

def scan(func, carry, xs, unroll):
    #length = len(jax.tree_util.tree_map(lambda x: x[0], xs))
    def inner(carry, x):
        x, rng = x
        carry, y = fun(rng, carry, x, create=False, modify=False)
        #post(carry), prior(y) = obs_step(self, prev_state, prev_action, embed, is_first)
        return carry, y
    carry, ys = jax.lax.scan(inner, carry, xs, length, False, unroll)
    return carry, ys

import gym

from common import Batch, InfoDict, Model, PRNGKey, MLP, Params, default_init
from dataclasses import dataclass, field

class EnsembleRSSM(nn.Module):
    training: bool 
    _deter: int = 1024
    _stoch: int = 32
    _classes: int = 32
    _unroll: bool = False
    _initial: str = 'learned'
    _unimix: float = 0.01
    _action_clip: float = 1.0
    _units: int = 640
    _ensemble: int = 5

    def setup(self):
        self.rssms = [RSSM(self.training) for i in range(self._ensemble)]

    def initial(self, bs):
        res = []
        for i in range(self._ensemble):
            res.append(self.rssms[i].initial(bs))
        return res

    def __call__(self, key, embed, action, is_first, state=None):
        bs = embed.shape[0]
        if self.training:
            assert embed.shape[1] == action.shape[1]
            return self.observe(key, embed, action, is_first, state)
        else:
            assert embed is None
            return self.imagine(action)

    def observe(self, key, embed, action, is_first, state):
        batch_size = embed.shape[0]
        if state is None:
            state = self.initial(batch_size)

        post, prior = [], []
        for i in range(self._ensemble):
            _post, _prior = self.rssms[i].observe(key, embed, action, is_first, state[i])
            post.append(_post)
            prior.append(_prior)
        post = {k: jnp.stack([_post[k] for _post in post], axis=0) for k in post[0]}
        prior = {k: jnp.stack([_prior[k] for _prior in prior], axis=0) for k in prior[0]}
        idx = jax.random.choice(key, self._ensemble, (1, batch_size))
        post = {k: jnp.take_along_axis(post[k], idx.reshape((1, batch_size,) + (1,)*(len(post[k].shape)-2)), axis=0)[0] for k in post}
        prior = {k: jnp.take_along_axis(prior[k], idx.reshape((1, batch_size,) + (1,)*(len(prior[k].shape)-2)), axis=0)[0] for k in prior}
        return post, prior 


class RSSM(nn.Module):
    training: bool 
    _deter: int = 1024
    _stoch: int = 32
    _classes: int = 32
    _unroll: bool = False
    _initial: str = 'learned'
    _unimix: float = 0.01
    _action_clip: float = 1.0
    _units: int = 640
    #_kw: dict = field(default_factory=lambda:{'units': 640, 'act': 'silu', 'norm': 'layer'}) 

    def setup(self):
        self.initial_state = self.param('initial', nn.initializers.constant(0), (self._deter))
        self.img_in = nn.Sequential([nn.Dense(self._units, kernel_init=default_init()),
                                     nn.LayerNorm(), nn.silu])
        self.img_gru = nn.Dense(3 * self._deter, kernel_init=default_init())
        self.img_out = nn.Sequential([nn.Dense(self._units, kernel_init=default_init()),
                                     nn.LayerNorm(), nn.silu])
        self.img_stats = nn.Dense(self._stoch * self._classes, kernel_init=default_init())
        self.obs_out = nn.Sequential([nn.Dense(self._units, kernel_init=default_init()),
                                     nn.LayerNorm(), nn.silu])
        self.obs_stats = nn.Dense(self._stoch * self._classes, kernel_init=default_init())

    def initial(self, bs):
        state = {}
        state['deter'] = jnp.repeat(jnp.tanh(self.initial_state)[None], bs, 0)
        state['stoch'] = self.get_stoch(cast(state['deter']))
        state['logit'] = jnp.zeros([bs, self._stoch, self._classes])
        return cast(state)

    def __call__(self, key, embed, action, is_first, state=None):
        bs = embed.shape[0]
        if state is None:
            state = self.initial(bs)

        if self.training:
            assert embed.shape[1] == action.shape[1]
            return self.observe(key, embed, action, is_first, state)
        else:
            assert embed is None
            return self.imagine(action)

    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        prior = scan(self.img_step, action, state, self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def _gru(self, x, deter):
        x = jnp.concatenate([deter, x], -1)
        x = self.img_gru(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def img_step(self, prev_state, key, prev_action):
        prev_stoch = prev_state['stoch']
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
            prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
        if self._classes:
            shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
            prev_stoch = prev_stoch.reshape(shape)
        #print(prev_action.shape, prev_stoch.shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
            prev_action = prev_action.reshape(shape)
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.img_in(x)
        x, deter = self._gru(x, prev_state['deter'])
        x = self.img_out(x)
        stats = self._img_stats(x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=key)
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return cast(prior)

    def observe(self, key, embed, action, is_first, state=None):
        if state is None:
            state = self.initial(action.shape[0])
        length = action.shape[1]
        swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
        step = lambda prev, inputs: self.obs_step(prev[0], prev[1], *inputs)
        inputs = (swap(action), swap(embed), swap(is_first))
        start = (key, state)
        post, prior = [], []
        for i in range(length):
            key, rng = jax.random.split(key)
            _post, _prior = self.obs_step(key, state, action[:, i], embed[:, i], is_first[:, i])
            post.append(_post)
            prior.append(_prior)
            state = _post

        post = {k: jnp.stack([post[i][k] for i in range(length)], axis=0) for k in post[0].keys()}
        prior = {k: jnp.stack([prior[i][k] for i in range(length)], axis=0) for k in prior[0].keys()} 
        #post, prior = jax.lax.scan(step, start, inputs, length, False, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def get_dist(self, state, argmax=False):
        logit = state['logit'].astype(jnp.float32)
        return tfd.Independent(jaxutils.OneHotDist(logit), 1)

    def _img_stats(self, x):
        # [sample(h_t) -> z_t]
        if self._classes:
            x = self.img_stats(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
        if self._unimix:
            probs = jax.nn.softmax(logit, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logit = jnp.log(probs)
        stats = {'logit': logit}
        return stats

    def _obs_stats(self, x):
        if self._classes:
            x = self.obs_stats(x)
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
        if self._unimix:
            probs = jax.nn.softmax(logit, -1)
            uniform = jnp.ones_like(probs) / probs.shape[-1]
            probs = (1 - self._unimix) * probs + self._unimix * uniform
            logit = jnp.log(probs)
        stats = {'logit': logit}
        return stats

    def get_stoch(self, deter):
        #x = self.get('img_out', Linear, **self._kw)(deter)
        x = self.img_out(deter)
        stats = self._img_stats(x)
        dist = self.get_dist(stats)
        return cast(dist.mode())

    def obs_step(self, key, prev_state, prev_action, embed, is_first):
        #print(key, prev_action.shape, embed.shape, is_first.shape)
        key, rng = jax.random.split(key)
        is_first = cast(is_first)
        prev_action = cast(prev_action)
        if self._action_clip > 0.0:
          prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
        prev_state, prev_action = jax.tree_util.tree_map(
            lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action)
        )
        prev_state = jax.tree_util.tree_map(
            lambda x, y: x + self._mask(y, is_first), prev_state, self.initial(len(is_first))
        )
        prior = self.img_step(prev_state, key, prev_action)

        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.obs_out(x); stats = self._obs_stats(x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=key)
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return cast(post), cast(prior)

    def _mask(self, value, mask):
        return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

class _RSSM(nn.Module):
  def initial(self, bs):
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          logit=jnp.zeros([bs, self._stoch, self._classes], f32),
          stoch=jnp.zeros([bs, self._stoch, self._classes], f32))
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          mean=jnp.zeros([bs, self._stoch], f32),
          std=jnp.ones([bs, self._stoch], f32),
          stoch=jnp.zeros([bs, self._stoch], f32))
    if self._initial == 'zeros':
      return cast(state)
    elif self._initial == 'learned':
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(cast(state['deter']))
      return cast(state)
    else:
      raise NotImplementedError(self._initial)

  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_dist(self, state, argmax=False):
    logit = state['logit'].astype(f32)
    return tfd.Independent(jaxutils.OneHotDist(logit), 1)

  def obs_step(self, prev_state, prev_action, embed, is_first):
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    prior = self.img_step(prev_state, prev_action)
    x = jnp.concatenate([prior['deter'], embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Linear, **self._kw)(x)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return cast(prior)

  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())

  def _gru(self, x, deter):
    x = jnp.concatenate([deter, x], -1)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

class RewardHead(nn.Module):
    num_bins: int = 255
    
    def setup(self):
        self.net = nn.Dense(self.num_bins)

    def __call__(self, x):
        x = self.net(x)
        print(x.shape)
        return jaxutils.DiscDist(x)

class ContHead(nn.Module):
    def setup(self):
        self.net = nn.Dense(1)

    def __call__(self, x):
        x = self.net(x)
        dist = tfd.Bernoulli(x)
        return tfd.Independent(dist, 1)

class DecoderHead(nn.Module):
    hidden_dims: Sequence[int]

    def setup(self):
        self.net = MLP(self.hidden_dims)

    def __call__(self, x):
        x = self.net(x)
        return jaxutils.SymlogDist(x, 1, 'mse', 'sum')

class WorldModel(nn.Module):
    hidden_dims: int
    obs_dim: int
    action_dim: int
    training: bool
    dropout_rate: Optional[float] = None
    _min: Optional[float] = -10.
    _max: Optional[float] = 0.5

    def setup(self):
        self.encoder = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)
        self.rssm = EnsembleRSSM(self.training)
        self.decoder = DecoderHead([self.hidden_dims[-1], self.obs_dim])
        #MLP([self.hidden_dims[-1], self.obs_dim], activate_final=False, dropout_rate=self.dropout_rate)
        self.reward_head = RewardHead(255)
        self.mask_head = ContHead()

    def __call__(self,
                 rng: PRNGKey,
                 observations: jnp.ndarray, # [N, L, d_obs]
                 actions: jnp.ndarray,      # [N, L, d_act]
                 is_first: jnp.ndarray,     # [N, L]
                 states: Dict[str, jnp.ndarray] = None,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
       
        if self.training:
            N, L = observations.shape[:2]
            print(observations.shape)
            embed = self.encoder(observations, training=training)
            post, prior = self.rssm(rng, embed, actions, jnp.zeros((N, L), dtype=bool), states)

            feats = jnp.concatenate([post['deter'].reshape((N, L, -1)),
                                     post['stoch'].reshape((N, L, -1)),], axis = 2)
            r_hat_dist = self.reward_head(feats); r_hat = r_hat_dist.mode()
            cont_hat_dist = self.mask_head(feats); done_hat = 1 - cont_hat_dist.mode()
            s_hat_dist = self.decoder(feats); s_hat = s_hat_dist

            info = {
                'post': post,
                'prior': prior,
                'reward': r_hat_dist,
                'cont': cont_hat_dist,
                's_hat': s_hat_dist
            }
            return s_hat, r_hat, done_hat, info 

        else:
            assert state is not None 
            N, L = observations.shape[:2]
            z = self.encoder(observations, training=training)
            post, prior = self.rssm(rng, z, actions, jnp.zeros((N, L), dtype=bool), states)

            feats = {**post, 'embed': embed}
            r_hat_dist = self.reward_head(feats); r_hat = r_hat_dist.mode()
            cont_hat_dist = self.mask_head(feats); done_hat = 1 - cont_hat_dist.mode()
            s_hat_dist = self.decoder(feats); s_hat = s_hat_dist.mode()

            return s_hat, r_hat, done_hat, {} 
