import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    lr = 3e-4
    config.actor_lr = lr
    config.alpha_lr = lr
    config.value_lr = lr
    config.critic_lr = lr

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 3.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    ## World Model
    config.num_models = 7
    config.num_elites = 5
    config.model_lr = 3e-4
    config.model_hidden_dims = (200, 200, 200, 200)

    return config
