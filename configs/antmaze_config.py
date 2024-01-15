import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 10.0
    config.dropout_rate = None

    config.tau = 0.005  # For soft target updates.

    ## World Model

    config.num_models = 7
    config.num_elites = 5
    config.model_lr = 3e-4
    config.model_hidden_dims = (256, 256, 256, 256)

    return config
