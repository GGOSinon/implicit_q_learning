import numpy as np

from gym.envs.mujoco import mujoco_env
from typing import Callable, List, Tuple, Dict

class BatchedMujocoOracleDynamics(object):
    def __init__(self, env_name: str) -> None:
        self.env = env

    def _set_state_from_obs(self, env:mujoco_env, obs:np.ndarray) -> None:
        if len(obs) == (self.env.model.nq + self.env.model.nv - 1):
            xpos = np.zeros(1)
            obs = np.concatenate([xpos, obs])
        qpos = obs[:self.env.model.nq]
        qvel = obs[self.env.model.nq:]
        self.env._elapsed_steps = 0
        self.env.set_state(qpos, qvel)

    def __call__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.set_state(obs)
        next_obs, rewards, masks = [], [], []
        for i in range(obs.shape[0]):
            self._set_state_from_obs(self.env[i], obs[i])

        next_obs, reward, terminal, _ = self.step(obs[i], action[i])
        return next_obs, rewards, masks

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        self.env.reset()
        self._set_state_from_obs(obs)
        next_obs, reward, terminal, info = self.env.step(action)
        return next_obs, reward, terminal, info

class MujocoOracleDynamics(object):
    def __init__(self, env: mujoco_env.MujocoEnv) -> None:
        self.env = env

    def _set_state_from_obs(self, obs:np.ndarray) -> None:
        if len(obs) == (self.env.model.nq + self.env.model.nv - 1):
            xpos = np.zeros(1)
            obs = np.concatenate([xpos, obs])
        qpos = obs[:self.env.model.nq]
        qvel = obs[self.env.model.nq:]
        self.env._elapsed_steps = 0
        self.env.set_state(qpos, qvel)

    def __call__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obs, rewards, masks = [], [], []
        for i in range(obs.shape[0]):
            _next_obs, _reward, _terminal, _ = self.step(obs[i], action[i])
            next_obs.append(_next_obs)
            rewards.append(_reward)
            masks.append(1 - float(_terminal))
        next_obs = np.array(next_obs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        return next_obs, rewards, masks

    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        self.env.reset()
        self._set_state_from_obs(obs)
        next_obs, reward, terminal, info = self.env.step(action)
        return next_obs, reward, terminal, info
