from typing import Dict, List

import flax.linen as nn
import gym
import numpy as np
import copy
import time
import os
import pickle as pkl
from tqdm import tqdm

def take_video(seed: int, agent: nn.Module, render_env: gym.Env, termination_fn) -> Dict[str, float]:
    observation = render_env.reset()
    images, q_values, states, actions, rewards = [], [], [observation], [], []
    while True:
        image = render_env.render(mode='rgb_array')
        images.append(image)
        action = agent.sample_actions(observation, temperature=0.0)[0]
        observation, reward, done, info = render_env.step(action)
        q1, q2 = agent.critic(observation[None], action[None])
        q_values.append(np.minimum(q1, q2)[0])
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
        if done:
            break
    images = np.stack(images, axis=0)
    q_values = np.stack(q_values, axis=0)
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    rewards = np.stack(rewards, axis=0)
    return images, q_values, (states, actions, rewards)

def evaluate(seed: int, agent: nn.Module, envs: gym.vector.VectorEnv, video_path: str, step:int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
     
    states, actions, rewards = [], [], []
    if isinstance(envs, list):
        observations, dones = [], []
        num_episodes = len(envs)
        s = time.time()
        for env in envs:
            observations.append(env.reset())
            dones.append(False)
            states.append([])
            actions.append([])
            rewards.append([])
        observations = np.array(observations)
        dones = np.array(dones, dtype=bool)
        for j in range(10000):
            if np.all(dones): break
            _actions = agent.sample_actions(observations, temperature=0.0)
            _actions = np.array(_actions)
            for i in range(len(envs)):
                if dones[i]: continue
                states[i].append(np.copy(observations[i]))
                actions[i].append(np.copy(_actions[i]))
                obs, reward, done, info = envs[i].step(_actions[i])
                observations[i] = obs
                rewards[i].append(reward)
                if done:
                    dones[i] = True
                    stats['return'].append(info['episode']['return'])
                    stats['length'].append(info['episode']['length'])

        for k, v in stats.items():
            stats[k] = np.mean(v)

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
    else:
        num_episodes = envs.num_envs
        s = time.time()
        terms = np.zeros(num_episodes, dtype=np.int32)
        if seed is None:
            observations, dones = envs.reset(), np.zeros(num_episodes, dtype=bool)
        else:
            observations, dones = envs.reset(seed=seed), np.zeros(num_episodes, dtype=bool)
        for j in range(10000):
            if np.all(terms > 0): break
            _actions = agent.sample_actions(observations, temperature=0.0)
            _actions = np.array(_actions)
            states.append(observations)
            actions.append(_actions)
            observations, _rewards, dones, infos = envs.step(actions)
            rewards.append(_rewards)
            stats['return'].append(_rewards)
            stats['length'].append(np.ones(num_episodes))
            for i in range(num_episodes): 
                if dones[i] and terms[i] == 0:
                    terms[i] = step

        for k, v in stats.items():
            tmp = []; v = np.array(v) 
            for i in range(num_episodes):
                tmp.append(np.sum(v[:terms[i], i]))
            stats[k] = np.mean(tmp)

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.stack(rewards, axis=0)
        states = np.concatenate([states[i, :terms[i]] for i in range(len(envs))], axis=0)
        actions = np.concatenate([actions[i, :terms[i]] for i in range(len(envs))], axis=0)
        rewards = np.concatenate([rewards[i, :terms[i]] for i in range(len(envs))], axis=0)

    q1, q2 = agent.critic(states, actions); q_values = np.minimum(q1, q2)
    trajectory = (states, actions, rewards)
    print("Saving to:", video_path, step)
    np.save(os.path.join(video_path, f"q_values_{step}.npz"), q_values)
    with open(os.path.join(video_path, f"traj_{step}.pkl"), 'wb') as F:
        pkl.dump(trajectory, F)
    return stats

def evaluate_single_env(agent: nn.Module, env: gym.Env,
                        num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)[0]
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
