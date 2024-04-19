import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import gym
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'returns_to_go'])

def expectile_loss(target, pred, expectile):
    weight = jnp.where(target > pred, expectile, (1 - expectile))
    diff = target - pred
    return weight * (diff ** 2)

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def inverse_sigmoid(x):
    return jnp.log(x) - jnp.log(1-x)

def symlog(x):
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]

class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    else:
      from dm_control import suite
      self._env = suite.load(domain, task, task_kwargs={'random':seed})
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def observation_space(self):
    return self.obs_space['image']

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  @property
  def action_space(self):
    return self.act_space['action']

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']
    reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action['action'])
      reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})




class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    use_norm: Optional[bool] = False
    use_symlog: Optional[bool] = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if self.use_symlog: x = symlog(x)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.use_norm: x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, 'Model']:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
