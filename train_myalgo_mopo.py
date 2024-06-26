import os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["NUM_INTRA_THREADS"] = '1'
os.environ["NUM_INTER_THREADS"] = '1'
from typing import Tuple

import cv2
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import jax
import jax.numpy as jnp
import subprocess
import pickle as pkl

from dynamics.termination_fns import get_termination_fn
import wandb
import wrappers
import orbax.checkpoint
from dataset_utils import D4RLDataset, NeoRLDataset, DMCDataset, split_into_trajectories, ReplayBuffer
from evaluation import evaluate, take_video
from algos.myalgo_mopo.learner import Learner
import common
#from algos.myalgo_dreamer.learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'antmaze-medium-play-v0', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Dynamics model load dir')
flags.DEFINE_string('save_dir', './tmp/EP/', 'Tensorboard logging dir.')
flags.DEFINE_string('wandb_key', '', 'Wandb key')
flags.DEFINE_string('dynamics', 'torch', 'Dynamics model')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('num_layers', 3, 'number of hidden layers')
flags.DEFINE_integer('layer_size', 256, 'layer size')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_float('discount', 0.99, 'discount')
flags.DEFINE_float('lamb', 0.95, 'lambda for GAE')
flags.DEFINE_float('cql_weight', None, 'CQL weight.')
flags.DEFINE_float('target_beta', None, 'Target cql beta for lagrange.')
flags.DEFINE_float('temp_explore', 1.0, 'Temperature for exploration.')
flags.DEFINE_float('expectile', None, 'Expectile for Q estimation')
flags.DEFINE_float('expectile_policy', 0.5, 'Expectile for V estimation')
#flags.DEFINE_float('sac_alpha', 0.2, 'SAC alpha.')
flags.DEFINE_float('model_batch_ratio', 0.95, 'Model-data batch ratio.')
flags.DEFINE_integer('rollout_batch_size', 50000, 'Rollout batch size.')
flags.DEFINE_integer('rollout_freq', 1000, 'Rollout batch size.')
flags.DEFINE_integer('rollout_length', 5, 'Rollout length.')
flags.DEFINE_integer('num_repeat', 1, 'Number of rollouts')
flags.DEFINE_integer('rollout_retain', 5, 'Rollout retain')
flags.DEFINE_integer('horizon_length', 5, 'Value estimation length.')
flags.DEFINE_integer('num_actor_updates', 1, 'Number of actor updates')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('T', 16, 'T')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('max_q_backup', False, 'Use max q backup')
flags.DEFINE_boolean('debug', False, 'Use max q backup')
flags.DEFINE_string('baseline', None, 'Use baseline')
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
                         seed, discount, model=None) -> Tuple[gym.Env, D4RLDataset]:

    is_neorl = env_name.split('-')[1] == 'v3'
    is_dmc = ('v2' not in env_name) and ('v3' not in env_name)
    if is_dmc:
        task, diff = env_name.split('-')
        env = common.DMC(task, 2, (64, 64), -1)
        env = common.NormalizeAction(env)
        data_path = f'../../v-d4rl/{task}/{diff}/64px'
        dataset = DMCDataset(data_path, FLAGS.T, model, discount)
        raw_dataset = dataset
    else:
        if is_neorl:
            import neorl
            task, version, data_type = tuple(env_name.split("-"))
            env = neorl.make(task+'-'+version)
            dataset = NeoRLDataset(env, data_type, discount)
        else:
            env = gym.make(env_name)
            dataset = D4RLDataset(env, discount)
        raw_dataset = env.get_dataset()
        env = wrappers.EpisodeMonitor(env)
        env = wrappers.SinglePrecision(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    env_name = FLAGS.env_name.lower()
    if 'antmaze' in env_name:
        #dataset.rewards -= 1.0
        #reward_scale, reward_bias = 1., -1.
        dataset.rewards = dataset.rewards * 10 - 5.
        reward_scale, reward_bias = 10., -5.
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in env_name or 'walker2d' in env_name
          or 'hopper' in env_name):
        if 'random' in env_name:
            reward_scale, reward_bias = 1., 0.
        else:
            reward_scale, reward_bias = normalize(dataset)
        #normalize(dataset)
    else:
        reward_scale, reward_bias = 1., 0.

    return env, raw_dataset, dataset, (reward_scale, reward_bias)

def get_normalized_score_neorl(x, env_name):
    if env_name == 'HalfCheetah':
        max_score = 12284
        min_score = -298
    if env_name == 'Hopper':
        max_score = 3294
        min_score = 5
    if env_name == 'Walker2d':
        max_score = 5143
        min_score = 1
    return (x - min_score) / (max_score - min_score)

def main(_):
    if FLAGS.debug is False:
        wandb.login(key=FLAGS.wandb_key)
        run = wandb.init(
        # Set the project where this run will be logged
        project="IQL",
            name=f"{FLAGS.env_name}_{FLAGS.seed}",
        # Track hyperparameters and run metadata
        config={
            "env_name": FLAGS.env_name,
            "seed": FLAGS.seed,
                "expectile": FLAGS.expectile,
                "expectile_policy": FLAGS.expectile_policy,
                "rollout_length": FLAGS.rollout_length,
                "horizon_length": FLAGS.horizon_length,
                "best": None,
                "num_updates": FLAGS.num_actor_updates,
                "num_repeat": FLAGS.num_repeat,
                "use_baseline": FLAGS.baseline,
                "model_batch_ratio": FLAGS.model_batch_ratio,
                "discount": FLAGS.discount,
                "rollout_retain": FLAGS.rollout_retain,
                "lamb": FLAGS.lamb,
        },
        )

    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                            str(FLAGS.seed)),
                                  write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    kwargs = dict(FLAGS.config)

    if ('v2' not in FLAGS.env_name) and ('v3' not in FLAGS.env_name):
        task, diff = FLAGS.env_name.split('-')
        env = common.DMC(task, 2, (64, 64), -1)
    else:
        if 'v3' in FLAGS.env_name:
            import neorl
            task, version, data_type = tuple(FLAGS.env_name.split("-"))
            env = neorl.make(task+'-'+version)
        else:
            env = gym.make(FLAGS.env_name)
   
    if FLAGS.dynamics == 'dreamerv2':
        obs_dim, action_dim = env.observation_space.shape, env.action_space.shape[-1]
        print(obs_dim, action_dim)
        from dynamics.dreamer_model import get_dreamer_wm
        if FLAGS.seed <= 3:
            model_path = f'../dreamerv2-EP/models/{task}/{diff}/{FLAGS.seed}/final_model.pkl'
        else:
            model_path = f'../dreamerv2-EP/models/{task}/{diff}/1/final_model.pkl'
        model_observe, model = get_dreamer_wm(model_path, obs_dim, action_dim, FLAGS.T)
        env, raw_dataset, dataset, reward_scaler = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.discount, model_observe)
        scaler = (jnp.zeros((1,1)), jnp.ones((1,1)))
    elif FLAGS.dynamics == 'torch': 
        obs_dim, action_dim = env.observation_space.shape[-1], env.action_space.shape[-1]
        print(obs_dim, action_dim)
        termination_fn = get_termination_fn(task=FLAGS.env_name)
        if 1 <= FLAGS.seed and FLAGS.seed <= 3:
            print("TESTING SEEDS!")
            model_path = os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', str(FLAGS.seed), env_name)
        else:
            model_path = os.path.join('../OfflineRL-Kit/models/dynamics-ensemble/', FLAGS.env_name)
        from dynamics.ensemble_model_learner import get_world_model
        env, raw_dataset, dataset, reward_scaler = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, FLAGS.discount)
        model, scaler = get_world_model(model_path, obs_dim, action_dim, reward_scaler, termination_fn)
    else:
        assert False, "Dynamics not given"
 
    if FLAGS.dynamics == 'dreamerv2':
        eval_envs = []
        for i in range(FLAGS.eval_episodes):
            env = common.DMC(task, 2, (64, 64), -1, seed=FLAGS.seed + i)
            env = common.NormalizeAction(env)
            env = wrappers.EpisodeMonitor(env) 
            #env = wrappers.SinglePrecision(env)   
            eval_envs.append(env)
    elif FLAGS.env_name.split('-')[1] == 'v3':
        name, version, _ = FLAGS.env_name.split('-')
        env_name = name + '-' + version
        eval_envs = gym.vector.make(env_name, FLAGS.eval_episodes, exclude_current_positions_from_observation=False)
        env.get_normalized_score = (lambda x: get_normalized_score_neorl(x, name))
    else:
        eval_envs = []
        for i in range(FLAGS.eval_episodes): 
            env = gym.make(FLAGS.env_name) 
            env = wrappers.EpisodeMonitor(env) 
            env = wrappers.SinglePrecision(env)   
            seed = FLAGS.seed + i 
            env.seed(seed)
            env.action_space.seed(seed) 
            env.observation_space.seed(seed) 
            eval_envs.append(env)

    data_batch = dataset.sample(FLAGS.batch_size)
    print(data_batch.observations.shape)
    print(data_batch.actions.shape)
    print("Finished loading dataset")
    agent = Learner(FLAGS.seed,
                    data_batch.observations,
                    data_batch.actions,
                    #np.repeat(env.observation_space.sample()[np.newaxis], FLAGS.batch_size, axis=0),
                    #np.repeat(env.action_space.sample()[np.newaxis], FLAGS.batch_size, axis=0),
                    max_steps=FLAGS.max_steps,
                    model=model,
                    env_name=FLAGS.env_name,
                    cql_weight=FLAGS.cql_weight,
                    target_beta=FLAGS.target_beta,
                    max_q_backup=FLAGS.max_q_backup,
                    scaler=scaler,
                    reward_scaler=reward_scaler,
                    horizon_length=FLAGS.horizon_length,
                    expectile=FLAGS.expectile,
                    expectile_policy=FLAGS.expectile_policy,
                    hidden_dims=tuple([FLAGS.layer_size for _ in range(FLAGS.num_layers)]),
                    discount=FLAGS.discount,
                    lamb=FLAGS.lamb,
                    num_actor_updates=FLAGS.num_actor_updates,
                    baseline=FLAGS.baseline,
                    num_repeat=FLAGS.num_repeat,
                    #sac_alpha=FLAGS.sac_alpha,
                    **kwargs)

    video_path = os.path.join(FLAGS.save_dir, 'videos', FLAGS.env_name, str(FLAGS.seed))
    model_path = os.path.join(FLAGS.save_dir, 'models', FLAGS.env_name, str(FLAGS.seed))
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    rollout_dataset = ReplayBuffer(data_batch.observations.shape[1], data_batch.actions.shape[1], capacity=FLAGS.rollout_retain*FLAGS.rollout_length*FLAGS.rollout_batch_size)

    #model_batch_size = int(FLAGS.batch_size * FLAGS.model_batch_ratio)
    #data_batch_size = FLAGS.batch_size - model_batch_size

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

        data_batch = dataset.sample(FLAGS.batch_size)
        model_batch = rollout_dataset.sample(FLAGS.batch_size)
        update_info = agent.update(data_batch, model_batch, FLAGS.model_batch_ratio)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                    run.log({f'training/{k}': v}, step=i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
                    run.log({f'training/{k}': wandb.Histogram(v)}, step=i)
            summary_writer.flush()

        '''
        if i % FLAGS.video_interval == 0:
            images, q_values, trajectory = take_video(FLAGS.seed, agent, env, agent.termination_fn)
            np.save(os.path.join(video_path, f"q_values_{i}.npz"), q_values)
            with open(os.path.join(video_path, f"traj_{i}.pkl"), 'wb') as F:
                pkl.dump(trajectory, F)
            video = cv2.VideoWriter(os.path.join(video_path, 'tmp.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 50, (images.shape[2], images.shape[1]), True)
            for j in range(images.shape[0]):
                video.write(images[j])
            video.release()
            subprocess.call(['ffmpeg', '-y', '-i', os.path.join(video_path, "tmp.mp4"), os.path.join(video_path, f"output_{i}.mp4")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        '''

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(FLAGS.seed, agent, eval_envs, video_path, i)
            if raw_dataset is not None:
                q_dataset = agent.critic(raw_dataset['observations'][::10], raw_dataset['actions'][::10])
                np.save(os.path.join(video_path, f"q_values_dataset_{i}.npz"), q_dataset)
            
            for k, v in eval_stats.items():
                #if k == 'return':
                #    v = env.get_normalized_score(v)
                #summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
                run.log({f'evaluation/average_{k}s': v}, step=i)
            #summary_writer.flush()

            eval_returns.append((i, env.get_normalized_score(eval_stats['return'])))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            params = {'actor': agent.actor.params,
                      'critic': agent.critic.params}
            with open(os.path.join(model_path, f'{FLAGS.seed}_{i}.pkl'), 'wb') as F:
                pkl.dump(params, F)

if __name__ == '__main__':
    app.run(main)
