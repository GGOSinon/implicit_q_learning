import collections
from typing import Optional, Tuple

import d4rl
import gym
import numpy as np
from tqdm import tqdm
from common import Batch, Model
import glob, os

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 returns_to_go: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.returns_to_go = returns_to_go
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     returns_to_go=self.returns_to_go[indx])

class NeoRLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 data_type: str,
                 discount: float = 1.0,
                 traj_num: int = 1000,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        train_data, _ = env.get_dataset(data_type=data_type, train_num=traj_num, need_val=False)
        dataset = {}
        dataset["observations"] = train_data["obs"]
        dataset["actions"] = train_data["action"]
        dataset["next_observations"] = train_data["next_obs"]
        dataset["rewards"] = train_data["reward"][:, 0]
        dataset["terminals"] = train_data["done"][:, 0]

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        returns_to_go = np.zeros_like(dataset['rewards'])
        #returns_to_go[-1] = dataset['rewards'][-1] 
        #for i in reversed(range(len(dones_float) - 1)):
        #    returns_to_go[i] = dataset['rewards'][i] + returns_to_go[i+1] * discount * (1.0 - dones_float[i])

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(np.float32),
                         returns_to_go=returns_to_go.astype(np.float32),
                         size=len(dataset['observations']))

class DMCDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 T: int,
                 model_observe: Model = None,
                 discount: float = 1.0,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,):

        dataset = []
        for filename in tqdm(sorted(glob.glob(os.path.join(data_path, '*.npz')))):
            with open(filename, 'rb') as F:
                episode = np.load(F)
                episode = {k: episode[k] for k in episode.keys()}
            dataset.append(episode)

        print(dataset[0].keys(), T)
        dataset = {k: np.concatenate([episode[k] for episode in dataset], axis=0) for k in dataset[0]}
        for k in dataset:
            print(k, dataset[k].shape)

        if clip_to_eps:
            lim = 1 - eps
            dataset['action'] = np.clip(dataset['action'], -lim, lim)

        #dataset['image'] = dataset['image'].transpose(0, 3, 1, 2)
        dones_float = 1 - dataset['is_terminal']
        for k in dataset:
            D = len(dataset[k].shape)
            dataset[k] = np.lib.stride_tricks.sliding_window_view(dataset[k], T, axis=0)
            print("S", k, dataset[k].shape)
            dataset[k] = dataset[k].transpose([0, D] + [i for i in range(1, D)])
            #if len(dataset[k].shape) == 3:
            #    dataset[k] = dataset[k].transpose((0, 2, 1))
            print("T", k, dataset[k].shape)

        returns_to_go = np.zeros_like(dataset['reward'])
        #returns_to_go[-1] = dataset['rewards'][-1] 
        #for i in reversed(range(len(dones_float) - 1)):
        #    returns_to_go[i] = dataset['rewards'][i] + returns_to_go[i+1] * discount * (1.0 - dones_float[i])

        self.indxs = np.where(np.all(np.logical_or(dataset['is_last'] == 0, dataset['is_terminal']), axis=1))[0]
        print("IMAGE", dataset['image'].min(), dataset['image'].max())
        if model_observe is None:
            states = dataset['image'][:-1].astype(np.float32)
            actions = dataset['action'][:-1].astype(np.float32)
            rewards = dataset['reward'][:-1].astype(np.float32)
            masks = 1.0 - dataset['is_terminal'][:-1].astype(np.float32)
            dones_float = dataset['is_last'][:-1].astype(np.float32)
            next_states = dataset['image'][1:].astype(np.float32)
            returns_to_go = returns_to_go.astype(np.float32)
            size=self.indxs.shape[0]

        else:
            import jax
            states, actions, rewards, masks, dones_float, next_states, size = [], [], [], [], [], [], 0
            rng = jax.random.PRNGKey(42)
            batch_size = 50; is_first = np.zeros((batch_size, T))
            for i in range(0, len(self.indxs), batch_size):
                idxs = self.indxs[i:i+batch_size]
                rng, key = jax.random.split(rng)
                _states = model_observe(key, dataset['image'][idxs], dataset['action'][idxs], is_first, None)
                _states = np.concatenate([_states['stoch'].reshape((batch_size, T, -1)), _states['deter']], axis=-1) 
                states.append(_states[:, :-1])
                actions.append(dataset['action'][idxs, :-1])
                rewards.append(dataset['reward'][idxs, :-1])
                masks.append(1.0 - dataset['is_terminal'][idxs, :-1])
                dones_float.append(dataset['is_last'][idxs, :-1])
                next_states.append(_states[:, 1:])
                size += (batch_size * (T-1))
                break
            states = np.concatenate(np.concatenate(states, axis=0), axis=0)
            actions = np.concatenate(np.concatenate(actions, axis=0), axis=0)
            rewards = np.concatenate(np.concatenate(rewards, axis=0), axis=0)
            masks = np.concatenate(np.concatenate(masks, axis=0), axis=0)
            dones_float = np.concatenate(np.concatenate(dones_float, axis=0), axis=0)
            next_states = np.concatenate(np.concatenate(next_states, axis=0), axis=0)
            self.indxs = np.arange(size)
            print("States", states.shape, states.max(), states.min())
            print("Actions", actions.shape)

        super().__init__(states, actions, rewards, masks, dones_float, next_states, returns_to_go, size)

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.choice(self.indxs, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     returns_to_go=self.returns_to_go[indx])

if __name__ == '__main__':
    dataset = DMCDataset('../../v-d4rl/walker_walk/medium/64px', 5)
    print(dataset)

class D4RLTimeDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 T: int,
                 discount: float = 1.0,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1
        dataset['dones_float'] = dones_float
        for k in dataset:
            D = len(dataset[k].shape)
            dataset[k] = np.lib.stride_tricks.sliding_window_view(dataset[k], T, axis=0)
            dataset[k] = np.transpose(dataset[k], [0] + [i%D+1 for i in range(1, D+1)])
            #print(k, dataset[k].shape)

        self.indxs = np.where(np.all(np.logical_or(dataset['dones_float'] == 0, dataset['terminals']), axis=1))[0]
        print(self.indxs)

        returns_to_go = np.zeros_like(dataset['rewards'])
        #returns_to_go[-1] = dataset['rewards'][-1] 
        #for i in reversed(range(len(dones_float) - 1)):
        #    returns_to_go[i] = dataset['rewards'][i] + returns_to_go[i+1] * discount * (1.0 - dones_float[i])

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dataset['dones_float'].astype(np.float32),
                         next_observations=dataset['next_observations'].astype(np.float32),
                         returns_to_go=returns_to_go.astype(np.float32),
                         size=len(self.indxs))

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.choice(self.indxs, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     returns_to_go=self.returns_to_go[indx])

class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 discount: float = 1.0,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        returns_to_go = np.zeros_like(dataset['rewards'])
        #returns_to_go[-1] = dataset['rewards'][-1] 
        #for i in reversed(range(len(dones_float) - 1)):
        #    returns_to_go[i] = dataset['rewards'][i] + returns_to_go[i+1] * discount * (1.0 - dones_float[i])

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(np.float32),
                         returns_to_go=returns_to_go.astype(np.float32),
                         size=len(dataset['observations']))

class ReplayBuffer(Dataset):
    def __init__(self, observation_dim: int, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, observation_dim), dtype=np.float32)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        returns_to_go = np.empty((capacity, ), dtype=np.float32)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         returns_to_go=returns_to_go,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[indices]
        self.returns_to_go[:num_samples] = dataset.returns_to_go[indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float, returns_to_go: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation
        self.returns_to_go[self.insert_index] = returns_to_go

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def insert_batch(self, observations: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, masks: np.ndarray, dones_float: np.ndarray,
                     next_observations: np.ndarray):
        batch_size = observations.shape[0]
        if self.insert_index + batch_size > self.capacity:
            p = self.capacity - self.insert_index
            self.insert_batch(observations[:p], actions[:p], rewards[:p], masks[:p], dones_float[:p], next_observations[:p])
            self.insert_batch(observations[p:], actions[p:], rewards[p:], masks[p:], dones_float[p:], next_observations[p:])
            return
        self.observations[self.insert_index:self.insert_index + batch_size] = observations
        self.actions[self.insert_index:self.insert_index + batch_size] = actions
        self.rewards[self.insert_index:self.insert_index + batch_size] = rewards
        self.masks[self.insert_index:self.insert_index + batch_size] = masks
        self.dones_float[self.insert_index:self.insert_index + batch_size] = dones_float
        self.next_observations[self.insert_index:self.insert_index + batch_size] = next_observations
        self.returns_to_go[self.insert_index:self.insert_index + batch_size] = None

        self.insert_index = (self.insert_index + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

class ReplayTimeBuffer(ReplayBuffer):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
            capacity: int, T: int):

        observations = np.empty((capacity, T, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, T, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, T), dtype=np.float32)
        masks = np.empty((capacity, T), dtype=np.float32)
        dones_float = np.empty((capacity, T), dtype=np.float32)
        next_observations = np.empty((capacity, T, *observation_space.shape),
                                     dtype=observation_space.dtype)
        returns_to_go = np.empty((capacity, T), dtype=np.float32)
        super(ReplayBuffer, self).__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         returns_to_go=returns_to_go,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity


