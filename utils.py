import torch
import collections


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def create_optim(config, network):
    """ Create the optimizer
    """
    if config.opt=='sgd':
        optim = torch.optim.SGD(network.parameters(),
                    lr=config.learning_rate,
                    momentum=0.9,
                    weight_decay=4e-4,
                    nesterov=False)
    elif config.opt=='adam' or config.opt=='sgdr':
        optim = torch.optim.Adam(network.parameters(),
                    lr=config.learning_rate,
                    weight_decay=4e-4
                    )
    elif config.opt=='rmsprop':
        optim = torch.optim.RMSprop(network.parameters(),
                    lr = config.learning_rate,
                    weight_decay = 1e-4)
    else:
        raise NotImplementedError

    return optim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





