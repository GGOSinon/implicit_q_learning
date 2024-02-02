import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import jax
import jax.numpy as jnp

import wandb
import wrappers
import orbax.checkpoint
from dataset_utils import D4RLDataset, split_into_trajectories, ReplayBuffer
from evaluation import evaluate
from algos.myalgo_mopo.learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'antmaze-medium-play-v0', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Dynamics model load dir')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('wandb_key', '', 'Wandb key')
flags.DEFINE_string('dynamics', 'torch', 'Dynamics model')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('num_layers', 3, 'number of hidden layers')
flags.DEFINE_integer('layer_size', 256, 'layer size')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('cql_weight', None, 'CQL weight.')
flags.DEFINE_float('target_beta', None, 'Target cql beta for lagrange.')
flags.DEFINE_float('temp_explore', 1.0, 'Temperature for exploration.')
flags.DEFINE_float('expectile', None, 'Expectile for Q estimation')
#flags.DEFINE_float('sac_alpha', 0.2, 'SAC alpha.')
flags.DEFINE_float('model_batch_ratio', 0.5, 'Model-data batch ratio.')
flags.DEFINE_integer('rollout_batch_size', 50000, 'Rollout batch size.')
flags.DEFINE_integer('rollout_freq', 1000, 'Rollout batch size.')
flags.DEFINE_integer('rollout_length', 5, 'Rollout length.')
flags.DEFINE_integer('rollout_retain', 5, 'Rollout retain')
flags.DEFINE_integer('horizon_length', 5, 'Value estimation length.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('max_q_backup', False, 'Use max q backup')
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

    scale = 1000.0 / (compute_returns(trajs[-1]) - compute_returns(trajs[0]))
    #scale = 1.
    dataset.rewards *= scale
    dataset.returns_to_go *= scale
    return scale, 0.


def make_env_and_dataset(env_name,
                         seed, discount) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env, discount)

    if 'antmaze' in FLAGS.env_name:
        #dataset.rewards -= 1.0
        dataset.rewards = dataset.rewards * 10 - 5.
        reward_scale, reward_bias = 10., -5.
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        reward_scale, reward_bias = normalize(dataset)
        #normalize(dataset)

    return env, dataset, (reward_scale, reward_bias)


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    kwargs = dict(FLAGS.config)

    env, dataset, reward_scaler = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, kwargs['discount'])
   
    eval_envs = gym.vector.make(FLAGS.env_name, FLAGS.eval_episodes)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    dynamics_name=FLAGS.dynamics,
                    env_name=FLAGS.env_name,
                    cql_weight=FLAGS.cql_weight,
                    target_beta=FLAGS.target_beta,
                    max_q_backup=FLAGS.max_q_backup,
                    reward_scaler=reward_scaler,
                    horizon_length=FLAGS.horizon_length,
                    expectile=FLAGS.expectile,
                    hidden_dims=tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)]),
                    #sac_alpha=FLAGS.sac_alpha,
                    **kwargs)

    if FLAGS.dynamics == 'torch':
        pass

    elif FLAGS.dynamics != 'oracle':
        if FLAGS.load_dir is None:
            file_dir = os.path.join('models', FLAGS.env_name, FLAGS.dynamics) 
        else:
            file_dir = FLAGS.load_dir
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(file_dir)
        agent.model = agent.model.replace(params = raw_restored['model'])

    rollout_dataset = ReplayBuffer(env.observation_space, env.action_space.shape[0], capacity=FLAGS.rollout_retain*FLAGS.rollout_length*FLAGS.rollout_batch_size)

    wandb.login(key=FLAGS.wandb_key)
    run = wandb.init(
	# Set the project where this run will be logged
	project="IQL",
        name=f"{FLAGS.env_name}_{FLAGS.seed}",
	# Track hyperparameters and run metadata
	config={
	    "env_name": FLAGS.env_name,
	    "seed": FLAGS.seed,
	},
    )

    model_batch_size = int(FLAGS.batch_size * FLAGS.model_batch_ratio)
    data_batch_size = FLAGS.batch_size - model_batch_size

    key = jax.random.PRNGKey(FLAGS.seed)
    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        if (i - 1) % FLAGS.rollout_freq == 0:
            key, rng = jax.random.split(key)
            data_batch = dataset.sample(FLAGS.rollout_batch_size)
            rollout = agent.rollout(rng, data_batch.observations, FLAGS.rollout_length, FLAGS.temp_explore)
            rollout_dataset.insert_batch(rollout['obss'], rollout['actions'], rollout['rewards'], rollout['masks'], 1 - rollout['masks'], rollout['next_obss'])

        data_batch = dataset.sample(data_batch_size)
        model_batch = rollout_dataset.sample(model_batch_size)
        update_info = agent.update(data_batch, model_batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                    run.log({f'training/{k}': v}, step=i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
                    run.log({f'training/{k}': wandb.Histogram(v)}, step=i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            #eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            eval_stats = evaluate(FLAGS.seed, agent, eval_envs)

            for k, v in eval_stats.items():
                if k == 'return':
                    v = env.get_normalized_score(v)
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
                run.log({f'evaluation/average_{k}s': v}, step=i)
            summary_writer.flush()

            eval_returns.append((i, env.get_normalized_score(eval_stats['return'])))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

if __name__ == '__main__':
    app.run(main)
