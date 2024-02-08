from typing import Dict, List

import flax.linen as nn
import gym
import numpy as np
import copy
import time
from tqdm import tqdm

def evaluate(seed: int, agent: nn.Module, envs: gym.vector.VectorEnv, render_env: gym.Env) -> Dict[str, float]:
    observation = render_env.reset()
    images = []
    while True:
        image = render_env.render(mode='rgb_array')
        images.append(image)
        actions = agent.sample_actions(observation, temperature=0.0)
        action = np.array(actions)[0]
        observation, reward, done, info = render_env.step(action)
        if done: break
    images = np.stack(images, axis=0)

    stats = {'return': [], 'length': []}
    
    num_episodes = envs.num_envs
    s = time.time()
    terms = np.zeros(num_episodes, dtype=np.int32)
    observations, dones = envs.reset(seed=seed), np.zeros(num_episodes, dtype=bool)
    step = 0
    while True:
        step += 1
        if np.all(terms > 0): break
        actions = agent.sample_actions(observations, temperature=0.0)
        actions = np.array(actions)
        observations, rewards, dones, infos = envs.step(actions)
        stats['return'].append(rewards)
        stats['length'].append(np.ones(num_episodes))
        for i in range(num_episodes): 
            if dones[i] and terms[i] == 0:
                terms[i] = step

    for k, v in stats.items():
        tmp = []; v = np.array(v) 
        for i in range(num_episodes):
            tmp.append(np.sum(v[:terms[i], i]))
        stats[k] = np.mean(tmp)

    return stats, images

def _evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
