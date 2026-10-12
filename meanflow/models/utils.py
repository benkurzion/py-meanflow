import torch


def noise_schedule(t, beta_min=0.1, beta_max=20):
    t = t.reshape(t.shape[0], 1, 1, 1)
    log_alpha_t = (-1/4) * torch.pow(t, 2) * (beta_max - beta_min) - (1/2) * t * beta_min
    alpha_t = torch.exp(log_alpha_t)
    sigma_t = torch.sqrt(1 - torch.pow(alpha_t, 2))
    return alpha_t, sigma_t


def sample_noisy_image(x_0, t):
    epsilon = torch.normal(mean=torch.zeros_like(x_0), std=1.0)
    alpha_t, sigma_t = noise_schedule(t)
    x_t = alpha_t * x_0 + sigma_t * epsilon
    return x_t, epsilon


def sample_t_diff(batch_size):
    return torch.rand(batch_size)
