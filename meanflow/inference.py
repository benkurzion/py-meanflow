import torch
import math
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


def stmd_inference_guided(model, n_inf, n_mf, samples_shape, device, A, y, zeta, max_rel_step=None, sigma_floor=0.02):
    """
    STMD inference (Alg. 2) + DPS-style measurement guidance.
    
    Args:
        A: frozen measurement operator --> eval mode, requires_grad_(False)
        y: measurement A(x_original), same batch size as the samples
        zeta: guidance scale applied to grad_{x_t} ||y - A(x0_hat)||
        max_rel_step: optional cap on ||guidance step|| as a fraction of sqrt(d) * sigma_{t'}
    """
    B = samples_shape[0]
    t = 1.0
    t_delta = 1 / n_inf
    s_delta = 1 / n_mf
    d = math.prod(samples_shape[1:])

    x_t = torch.randn(samples_shape, device=device)

    for k in range(n_inf):
        # 1) Differentiable sample x0 ~ p_theta(x0 | x_t)
        x_t = x_t.detach().requires_grad_(True)
        with torch.enable_grad():
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

            # per-sample residual norm; grad of the sum = per-sample grads (samples are independent)
            res_norm = (y - A(x_0_hat)).flatten(1).norm(dim=1)
            grad = torch.autograd.grad(res_norm.sum(), x_t)[0]

        # 2) Unguided stochastic transition x_t -> x_{t'} (Eq. 19)
        with torch.no_grad():
            x_0_hat = x_0_hat.detach()
            x_t = x_t.detach()

            alpha_t, sigma_t = noise_schedule(t=torch.tensor([t], device=device))
            alpha_tp, sigma_tp = noise_schedule(t=torch.tensor([t - t_delta], device=device))

            mu = (
                (alpha_tp**2 - alpha_t**2) / (alpha_tp * sigma_t**2) * x_0_hat
                + (alpha_t * sigma_tp**2) / (alpha_tp * sigma_t**2) * x_t
            )
            var = (sigma_tp**2 * (alpha_tp**2 - alpha_t**2)) / (alpha_tp**2 * sigma_t**2)
            x_prev = mu + torch.sqrt(var) * torch.randn(samples_shape, device=device)

            # 3) DPS update, applied after the transition
            step = zeta * grad
            if max_rel_step is not None:
                limit = max_rel_step * math.sqrt(d) * max(sigma_tp.item(), sigma_floor)
                step_norm = step.flatten(1).norm(dim=1).clamp_min(1e-12)
                scale = (limit / step_norm).clamp(max=1.0)
                step = step * scale.view(-1, 1, 1, 1)

            x_t = x_prev - step
            t = t - t_delta

    return x_t  # t' = 0 here, so this is the guided x_0