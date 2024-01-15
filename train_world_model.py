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
from dataset_utils import Dataset, D4RLDataset, split_into_trajectories
from dynamics.ensemble_model_learner import Learner
from flax.training import checkpoints, orbax_utils
import orbax

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
#flags.DEFINE_string('env_name', 'antmaze-medium-play-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 2048, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(100000), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0

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
        dataset.size - P)
    test_dataset = Dataset(
        dataset.observations[test_idxs],
        dataset.actions[test_idxs],
        dataset.rewards[test_idxs],
        dataset.masks[test_idxs],
        dataset.dones_float[test_idxs],
        dataset.next_observations[test_idxs],
        P)
    return train_dataset, test_dataset

def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)
    train_dataset, test_dataset = split_dataset(dataset)

    return env, train_dataset, test_dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    np.random.seed(FLAGS.seed)

    env, train_dataset, test_dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    valid_losses = []
    pbar = tqdm.tqdm(range(0, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm)

    ckpt = {'model': agent.model.params}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('./tmp/flax_ckpt/orbax/single_save/', ckpt, save_args=save_args, force=True)

    lossD, lossR, lossM = [], [], []
    for i in pbar:
        batch = train_dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)
        lossD.append(float(update_info['lossD']))
        lossR.append(float(update_info['lossR']))
        lossM.append(float(update_info['lossM']))
        pbar.set_postfix({'lossD': np.mean(lossD), 'lossR': np.mean(lossR), 'lossM': np.mean(lossM)})
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
            eval_stats = agent.evaluate(test_dataset, FLAGS.batch_size * 8)

            #for k, v in eval_stats.items():
            #    summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            #summary_writer.flush()

            valid_losses.append((i, eval_stats['loss']))
            print(eval_stats)

            ckpt = {'model': agent.model.params}
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save('./tmp/flax_ckpt/orbax/single_save/', ckpt, save_args=save_args, force=True)


if __name__ == '__main__':
    app.run(main)

