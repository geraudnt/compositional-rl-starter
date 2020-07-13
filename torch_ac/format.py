import torch

def default_preprocess_obs_goals(obs_goals, device=None):
    return torch.tensor(obs_goals, device=device)