import torch
from models.utils import noise_schedule


@torch.no_grad()
def stmd_inference(model, n_inf, n_mf, samples_shape, device):
    """
    STMD Inference — Algorithm 2 from the paper.

    Args:
        model: trained SongUNet
        n_inf: number of outer diffusion steps
        n_mf: number of inner mean flow steps
        samples_shape: (B, C, H, W)
        device: torch device
    """
    B = samples_shape[0]
    t = 1.0
    t_delta = 1 / n_inf
    s_delta = 1 / n_mf

    x_t = torch.randn(samples_shape, device=device)

    for k in range(n_inf):
        s = 1.0
        z_s = torch.randn(samples_shape, device=device)
        for i in range(n_mf):
            time_steps = (
                torch.full((B,), s, device=device),
                torch.full((B,), s_delta, device=device),
                torch.full((B,), t, device=device),
            )
            z_s = z_s - s_delta * model(torch.cat((z_s, x_t), dim=1), time_steps)
            s = s - s_delta

        x_0_hat = z_s

        alpha_t, sigma_t = noise_schedule(t=torch.tensor([t], device=device))
        alpha_t_prime, sigma_t_prime = noise_schedule(t=torch.tensor([t - t_delta], device=device))

        mu = (
            (alpha_t_prime ** 2 - alpha_t ** 2) / (alpha_t_prime * sigma_t ** 2) * x_0_hat
            + (alpha_t * sigma_t_prime ** 2) / (alpha_t_prime * sigma_t ** 2) * x_t
        )
        var = (sigma_t_prime ** 2 * (alpha_t_prime ** 2 - alpha_t ** 2)) / (alpha_t_prime ** 2 * sigma_t ** 2)

        x_t = mu + torch.sqrt(var) * torch.randn(samples_shape, device=device)
        t = t - t_delta

    return x_0_hat
