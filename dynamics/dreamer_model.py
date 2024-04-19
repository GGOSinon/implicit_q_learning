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
from dynamics.dreamer_nets import Encoder, DecoderHead, RewardHead, RSSM
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = lambda x: jax.tree_util.tree_map(lambda _x: _x.astype(jnp.float32), x)

from common import Batch, InfoDict, Model, PRNGKey, MLP, Params, default_init
from dataclasses import dataclass, field

class WorldModelObserve(nn.Module):
    #hidden_dims: int
    #obs_dim: int
    #action_dim: int
    dropout_rate: Optional[float] = None
    _min: Optional[float] = -10.
    _max: Optional[float] = 0.5

    def setup(self):
        self.encoder = Encoder()
        self.rssm = RSSM(True)
        #self.decoder = DecoderHead()
        #MLP([self.hidden_dims[-1], self.obs_dim], activate_final=False, dropout_rate=self.dropout_rate)
        self.reward_head = RewardHead()
        #self.mask_head = ContHead()

    def __call__(self,
                 rng: PRNGKey,
                 observations: jnp.ndarray, # [N, L, d_obs]
                 actions: jnp.ndarray,      # [N, L, d_act]
                 is_first: jnp.ndarray,     # [N, L]
                 states: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
       
        assert states is None
        N, L = observations.shape[:2]
        print(observations.shape)
        embed = self.encoder(observations)
        #jax.debug.print('EMBED {x}', x=embed)
        print(embed.shape, actions.shape)
        post, prior = self.rssm(rng, embed, actions, jnp.zeros((N, L), dtype=bool), None)
        return post

class WorldModelImagine(nn.Module):
    #hidden_dims: int
    #obs_dim: int
    #action_dim: int
    dropout_rate: Optional[float] = None
    _min: Optional[float] = -10.
    _max: Optional[float] = 0.5

    def setup(self):
        self.encoder = Encoder()
        self.rssm = RSSM(False)
        #self.decoder = DecoderHead()
        #MLP([self.hidden_dims[-1], self.obs_dim], activate_final=False, dropout_rate=self.dropout_rate)
        self.reward_head = RewardHead()
        #self.mask_head = ContHead()

    def __call__(self,
                 rng: PRNGKey,
                 states: jnp.ndarray, # [N, d_obs]
                 actions: jnp.ndarray,  # [N, d_act]
                 )-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
       
        batch_shape = actions.shape[:-1]; _stoch, _classes = self.rssm._stoch, self.rssm._classes
        states = {'stoch': states[..., :_stoch*_classes].reshape((*batch_shape, _stoch, _classes)),
                  'deter': states[..., _stoch*_classes:]}
        prior = self.rssm(rng, None, actions, None, states)
        #jax.debug.print('PRIOR {x}', x=prior)

        feats = jnp.concatenate([prior['stoch'].reshape((*batch_shape, -1)),
                                 prior['deter'].reshape((*batch_shape, -1)),], axis=-1)
        r_hat_dist = self.reward_head(feats); r_hat = r_hat_dist.mode().squeeze(-1)
        done_hat = jnp.zeros_like(r_hat)
        s_hat = feats
        #cont_hat_dist = self.mask_head(feats); done_hat = 1 - jax.nn.sigmoid(cont_hat_dist)
        #s_hat_dist = self.decoder(feats); s_hat = s_hat_dist.mode()

        #print(r_hat.shape, done_hat.shape, s_hat.shape)
        return s_hat, r_hat, done_hat, {}


def get_dreamer_wm(ckpt_path, observation_dim, action_dim, T):
    model_observe_def = WorldModelObserve()#(200, 200, 200), observation_dim, action_dim)
    model_imagine_def = WorldModelImagine()#model_observe_def.rssm.state_dim, action_dim)

    rng = jax.random.PRNGKey(42)
    observation = np.ones((1, T, *observation_dim))
    action = np.ones((1, T, action_dim))
    is_first = np.zeros((1, T))
    model_observe = Model.create(model_observe_def, inputs=[rng, rng, observation, action, is_first, None])
    states = model_observe(rng, observation, action, is_first, None)

    states = np.concatenate([states['stoch'].reshape((T, 1024)), states['deter'].reshape((T, 200))], axis=-1)
    action = np.ones((T, action_dim))
    model_imagine = Model.create(model_imagine_def, inputs=[rng, rng, states, action])
    #import pprint
    #pprint.pprint(jax.tree_util.tree_map(jnp.shape, model_observe.params))
    #pprint.pprint(jax.tree_util.tree_map(jnp.shape, model_imagine.params))
    
    import pickle as pkl
    with open(ckpt_path, 'rb') as F:
        ckpt = pkl.load(F)

    encoder = {}
    for i in range(4):
        encoder[f'conv{i+1}'] = {'kernel': 2*i+1, 'bias': 2*i+2}

    decoder = {}
    for i in range(4):
        decoder[f'conv{i+1}'] = {'kernel': 2*i+9, 'bias': 2*i+10}
    decoder[f'mlp1'] = {'kernel': 17, 'bias': 18}

    reward_head = {}
    for i in range(4):
        reward_head[f'mlp{i+1}'] = {'kernel': 2*i+19, 'bias': 2*i+20}
    reward_head['net'] = {'kernel': 27, 'bias': 28}

    rssm = {}
    rssm['img_gru'] = {'layers_0': {'kernel': 29, 'bias': 30}, 'layers_1': {'scale': 31, 'bias': 32}}
    rssm['img_in'] = {'layers_0': {'kernel': 47, 'bias': 48}}
    for i in range(7):
        rssm[f'img_stats_{i}'] = {'kernel': 2*i+33, 'bias': 2*i+34}
        rssm[f'img_out_{i}'] = {'layers_0': {'kernel': 2*i+49, 'bias': 2*i+50}}
    rssm[f'obs_stats'] = {'kernel': 63, 'bias': 64}
    rssm[f'obs_out'] = {'layers_0': {'kernel': 65, 'bias': 66}}

    jax_ckpt = {'encoder': encoder, 'decoder': decoder, 'reward_head': reward_head, 'rssm': rssm}
    jax_ckpt = jax.tree_util.tree_map(lambda x: ckpt[x], jax_ckpt)
    for i in range(4):
        jax_ckpt['decoder'][f'conv{i+1}']['kernel'] = jax_ckpt['decoder'][f'conv{i+1}']['kernel'].transpose((0, 1, 3, 2))

    model_observe_ckpt = {'encoder': jax_ckpt['encoder'], 'rssm': jax_ckpt['rssm']}
    model_imagine_ckpt = {'decoder': jax_ckpt['decoder'], 'rssm': jax_ckpt['rssm'], 'reward_head': jax_ckpt['reward_head']}

    import pprint
    pprint.pprint(jax.tree_util.tree_map(jnp.shape, model_observe_ckpt))

    model_observe = model_observe.replace(params = model_observe_ckpt)
    model_imagine = model_imagine.replace(params = model_imagine_ckpt)

    #for i in range(len(ckpt)):
    #    print(i, ckpt[i].shape)
    #    #for k in ckpt[i].keys():
    #    #    print(k, ckpt[i][k])
    return model_observe, model_imagine

if __name__ == '__main__':

    observation_dim = (64, 64, 3)
    action_dim = 6
    T = 5

    ckpt_path = '../dreamerv2-EP/models/walker_walk/medium/1/final_model.pkl'
    model_observe, model_imagine = get_dreamer_wm(ckpt_path, observation_dim, action_dim, T)

    states = model_observe(rng, observation, action, is_first, None)
    print("WFWEFWFFWFWEF")
    print(states)

    print("NEXT")
    states = {k: np.ones_like(v) for k, v in states.items()}
    next_states, rewards, done, info = model_imagine(rng, None, action, None, states)

    print(next_states, rewards, done, info)
