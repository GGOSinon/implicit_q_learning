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
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from model_learner import Learner
from flax.training import checkpoints, orbax_utils
import orbax

FLAGS = flags.FLAGS

#flags.DEFINE_string('env_name', 'antmaze-large-play-v0', 'Environment name.')
flags.DEFINE_string('env_name', 'antmaze-medium-play-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
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

    return env, dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)

    eval_returns = []
    pbar = tqdm.tqdm(range(0, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm)

    ckpt = {'model': agent.model.params}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('./tmp/flax_ckpt/orbax/single_save/', ckpt, save_args=save_args, force=True)

    lossD, lossR, lossP = [], [], []
    for i in pbar:
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)
        lossD.append(float(update_info['lossD']))
        lossR.append(float(update_info['lossR']))
        lossP.append(float(update_info['lossP']))
        if i % FLAGS.log_interval == 0:
            #print(type(lossD[0]), lossD[0])
            #lossD = jnp.stack(lossD, axis=0); lossR = jnp.stack(lossR, axis=0); lossP = jnp.stack(lossP, axis=0)
            lossD = np.mean(lossD); lossR = np.mean(lossR); lossP = np.mean(lossP)
            #lossD = jnp.mean(lossD); lossR = jnp.mean(lossR); lossP = jnp.mean(lossP)
            pbar.set_postfix({'lossD': lossD, 'lossR': lossR, 'lossP': lossP})
            lossD, lossR, lossP = [], [], []
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            '''
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            '''
            ckpt = {'model': agent.model.params}
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save('./tmp/flax_ckpt/orbax/single_save/', ckpt, save_args=save_args, force=True)


if __name__ == '__main__':
    app.run(main)

