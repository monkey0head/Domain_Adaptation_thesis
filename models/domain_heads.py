import torch.nn as nn

import configs.dann_config as dann_config


def get_domain_head(domain_input_len):
    if dann_config.DOMAIN_HEAD == "vanilla_dann":
        return vanilla_dann_domain_head(domain_input_len)
    elif dann_config.DOMAIN_HEAD == "dropout_dann":
        return dropout_dann_domain_head(domain_input_len)
    elif dann_config.DOMAIN_HEAD == "mnist_dann":
        return mnist_dann_domain_head(domain_input_len)
    else:
        raise RuntimeError("head %s does not exist" % dann_config.DOMAIN_HEAD)


def vanilla_dann_domain_head(domain_input_len):
    return nn.Sequential(
        nn.Linear(domain_input_len, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1),
    )


def dropout_dann_domain_head(domain_input_len):
    return nn.Sequential(
        nn.Linear(domain_input_len, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1),
    )


def mnist_dann_domain_head(domain_input_len):
    return nn.Sequential(
        nn.Linear(domain_input_len, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, 1),
    )
