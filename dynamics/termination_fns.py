import jax.numpy as np

def obs_unnormalization(termination_fn, obs_mean, obs_std):
    def thunk(obs, act, next_obs):
        obs = obs*obs_std + obs_mean
        next_obs = next_obs*obs_std + obs_mean
        return termination_fn(obs, act, next_obs)
    return thunk

def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
    done = ~not_done
    done = done[:, None]
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_antangle(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_antmaze(obs, act, next_obs, center, radius):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    done = ((next_obs[:, :2] - center[None, :]) ** 2).sum(axis=1) < radius
    done = done[:,None]
    return done

def termination_fn_ant(obs, act, next_obs):
    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                * (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_point2denv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_point2dwallenv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_pendulum(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros((len(obs), 1))
    return done

def termination_fn_humanoid(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:,0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:,None]
    return done

def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    obj_pos = next_obs[:, 24:27]
    done = obj_pos[:, 2] < 0.075

    done = done[:,None]
    return done

def terminaltion_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])

    done = done[:, None]
    return done

def termination_fn_neorl_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])
    done = done[:, None]
    return done

def termination_fn_neorl_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:, 1:2]
    angle = next_obs[:, 2:3]
    state = next_obs[:, 3:]

    min_state, max_state = (-100.0, 100.0)
    min_z, max_z = (0.7, float('inf'))
    min_angle, max_angle = (-0.2, 0.2)

    healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
    healthy_z = np.logical_and(min_z < z, z < max_z)
    healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)

    is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)

    done = np.logical_not(is_healthy).reshape(-1, 1)
    return done

def termination_fn_neorl_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    min_z, max_z = (0.8, 2.0)
    min_angle, max_angle = (-1.0, 1.0)
    min_state, max_state = (-100.0, 100.0)
    
    z = next_obs[:, 1:2]
    angle = next_obs[:, 2:3]
    state = next_obs[:, 3:]
    
    healthy_state = np.all(np.logical_and(min_state < state, state < max_state), axis=-1, keepdims=True)
    healthy_z = np.logical_and(min_z < z, z < max_z)
    healthy_angle = np.logical_and(min_angle < angle, angle < max_angle)
    is_healthy = np.logical_and(np.logical_and(healthy_state, healthy_z), healthy_angle)
    done = np.logical_not(is_healthy).reshape(-1, 1)
    return done

def termination_fn_kitchen(obs, act, next_obs):
    done = np.array([False] * obs.shape[0])
    done = done[:, None]
    return done

def get_termination_fn(task):
    if 'halfcheetahvel' in task:
        return termination_fn_halfcheetahveljump
    elif 'halfcheetah' in task:
        return termination_fn_halfcheetah
    elif 'hopper' in task:
        return termination_fn_hopper
    elif 'antangle' in task:
        return termination_fn_antangle
    elif 'antmaze' in task:
        import d4rl
        import gym
        env = gym.make(task)
        dataset = d4rl.qlearning_dataset(env)
        pos = dataset['observations'][dataset['terminals']][:, :2]
        cond = np.abs(pos - pos.mean(axis=0, keepdims=True)) < 2*pos.std(axis=0, keepdims=True)
        cond = np.logical_and(cond[:, 0], cond[:, 1])
        center = pos[cond].mean(axis=0); radius = ((pos[cond]-center)**2).sum(axis=1).max()
        print(task, center, radius)
        return lambda obs, action, next_obs: termination_fn_antmaze(obs, action, next_obs, center, radius)
    elif 'ant' in task:
        return termination_fn_ant
    elif 'walker2d' in task:
        return termination_fn_walker2d
    elif 'point2denv' in task:
        return termination_fn_point2denv
    elif 'point2dwallenv' in task:
        return termination_fn_point2dwallenv
    elif 'pendulum' in task:
        return termination_fn_pendulum
    elif 'humanoid' in task:
        return termination_fn_humanoid
    elif 'pen' in task:
        return termination_fn_pen
    elif 'door' in task:
        return terminaltion_fn_door
    elif 'HalfCheetah' in task:
        return termination_fn_neorl_halfcheetah
    elif 'Hopper' in task:
        return termination_fn_neorl_hopper
    elif 'Walker2d' in task:
        return termination_fn_neorl_walker2d
    elif 'kitchen' in task:
        return termination_fn_kitchen
    else:
        raise np.zeros
