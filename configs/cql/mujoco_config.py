import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    lr = 3e-4
    config.actor_lr = 3e-5
    config.alpha_lr = 3e-5
    config.value_lr = lr
    config.critic_lr = lr
    config.opt_decay_schedule = None

    config.hidden_dims = (256, 256, 256)

    config.discount = 0.99

    config.temperature = 1.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    return config
