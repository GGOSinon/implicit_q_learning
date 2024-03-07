import os
from typing import Tuple

import gym
import numpy as np
import jax.numpy as jnp
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers

from dataset_utils import Dataset, D4RLTimeDataset, split_into_trajectories
from dynamics.dreamer_learner import Learner
from flax.training import checkpoints, orbax_utils
import orbax

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
#flags.DEFINE_string('env_name', 'antmaze-medium-play-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 16, 'Mini batch size.')
flags.DEFINE_integer('time_size', 4, 'Time length.')
flags.DEFINE_integer('max_steps', int(100000), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def split_dataset(dataset, val_ratio = 0.2):
    P = int(dataset.size * val_ratio)
    idxs = np.random.permutation(list(range(dataset.size)))
    train_idxs, test_idxs = idxs[P:], idxs[:P]
    train_dataset = Dataset(
        dataset.observations[train_idxs],
        dataset.actions[train_idxs],
        dataset.rewards[train_idxs],
        dataset.masks[train_idxs],
        dataset.dones_float[train_idxs],
        dataset.next_observations[train_idxs],
        np.zeros_like(dataset.dones_float[train_idxs]),
        dataset.size - P)
    test_dataset = Dataset(
        dataset.observations[test_idxs],
        dataset.actions[test_idxs],
        dataset.rewards[test_idxs],
        dataset.masks[test_idxs],
        dataset.dones_float[test_idxs],
        dataset.next_observations[test_idxs],
        np.zeros_like(dataset.dones_float[test_idxs]),
        P)
    return train_dataset, test_dataset

def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLTimeDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLTimeDataset(env, FLAGS.time_size)

    train_dataset, test_dataset = split_dataset(dataset)

    return env, train_dataset, test_dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)

    env = gym.make(FLAGS.env_name)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    FLAGS.env_name,
                    FLAGS.batch_size,
                    max_steps=FLAGS.max_steps,
                    T=FLAGS.time_size,
                    **kwargs)

    env, train_dataset, test_dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    ckpt = agent.model.params
    print(ckpt.keys())

    valid_losses = []
    pbar = tqdm.tqdm(range(0, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm)

    lossD, lossR, lossP, lossM = [], [], [], []
    for i in pbar:
        batch = train_dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)
        lossD.append(float(update_info['lossD']))
        lossR.append(float(update_info['lossR']))
        lossP.append(float(update_info['lossP']))
        lossM.append(float(update_info['lossM']))
        pbar.set_postfix({'lossD': np.mean(lossD), 'lossR': np.mean(lossR), 'lossP': np.mean(lossP), 'lossM': np.mean(lossM)})
        if i % FLAGS.log_interval == 0:
            #print(type(lossD[0]), lossD[0])
            #lossD = jnp.stack(lossD, axis=0); lossR = jnp.stack(lossR, axis=0); lossP = jnp.stack(lossP, axis=0)
            #lossD = jnp.mean(lossD); lossR = jnp.mean(lossR); lossP = jnp.mean(lossP)
            lossD, lossR, lossM = [], [], []
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            lossDT, lossRT, lossPT, lossMT = [], [], [], []
            for _ in range(100):
                batch = test_dataset.sample(FLAGS.batch_size)
                update_info = agent.evaluate(batch)
                lossDT.append(float(update_info['lossD']))
                lossRT.append(float(update_info['lossR']))
                lossPT.append(float(update_info['lossP']))
                lossMT.append(float(update_info['lossM']))
                #lossDT.append(float(update_info['prior_ent_mean'] + update_info['post_ent_mean']))
                #lossRT.append(float(update_info['reward_loss_mean']))
                #lossPT.append(float(update_info['rep_loss_mean']))
                #lossMT.append(float(update_info['cont_pos_acc']))

            eval_stats = {'lossD': np.mean(lossDT), 'lossR': np.mean(lossRT), 'lossP': np.mean(lossPT), 'accM': np.mean(lossMT)}
            valid_losses.append((i, eval_stats))
            print(eval_stats)

            from flax.training import orbax_utils
            ckpt = agent.model.params
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(f'models/dreamer/{FLAGS.env_name}', ckpt, save_args=save_args)


if __name__ == '__main__':
    app.run(main)

