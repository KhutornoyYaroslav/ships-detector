import torch


def make_optimizer(lr:float, model:torch.nn.Module):
    return torch.optim.Adam(model.parameters(), lr=lr)


def make_lr_scheduler(cfg, optimizer):
    lambda_ = lambda epoch: cfg.SOLVER.LR_LAMBDA ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_)

    return scheduler