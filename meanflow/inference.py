import torch
import math
from models.utils import noise_schedule


@torch.no_grad()
def stmd_inference(model, n_inf, n_mf, samples_shape, device):
    """
    STMD Inference — Algorithm 2 from the paper.
    NOT guided

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

    return x_t


def stmd_inference_guided_new(model, n_inf, n_mf, samples_shape, device, A, y, zeta, rescale_for_A=True, max_rel_step=None, sigma_floor=0.02):
    """
    Per outer iteration:
      x_0_hat = mean-flow sample conditioned on x_t
      x_t_flow = bridge(x_0_hat, x_t)
      x_t_flow_guided = x_t_flow - zeta * grad_{x_t_flow} ||y - A(.)||

    rescale_for_A: feed A(x_t_flow / alpha_{t'}) so the signal part is in
                   data range (x_{t'} = alpha_{t'} x_0 + sigma_{t'} eps).
                   At t' = 0, alpha = 1 and this is a no-op.
    """
    d = math.prod(samples_shape[1:])
    B = samples_shape[0]
    t = 1.0
    t_delta = 1 / n_inf
    s_delta = 1 / n_mf
    
    x_t = torch.randn(samples_shape, device=device)

    for k in range(n_inf):
        with torch.no_grad():
            # mean-flow pipeline -> x_0_hat
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

            # unguided bridge: (x_t, x_0_hat) -> x_t_flow
            alpha_t, sigma_t = noise_schedule(t=torch.tensor([t], device=device))
            alpha_t_prime, sigma_t_prime = noise_schedule(t=torch.tensor([t - t_delta], device=device))

            mu = (
                (alpha_t_prime**2 - alpha_t**2) / (alpha_t_prime * sigma_t**2) * x_0_hat
                + (alpha_t * sigma_t_prime**2) / (alpha_t_prime * sigma_t**2) * x_t
            )
            var = (sigma_t_prime**2 * (alpha_t_prime**2 - alpha_t**2)) / (alpha_t_prime**2 * sigma_t**2)
            x_t_flow = mu + torch.sqrt(var) * torch.randn(samples_shape, device=device)

        # measurement gradient, evaluated on x_t_flow
        x_flow = x_t_flow.detach().requires_grad_(True)
        with torch.enable_grad():
            x_for_A = x_flow / alpha_t_prime if rescale_for_A else x_flow
            res_norm = (y - A(x_for_A)).flatten(1).norm(dim=1)
            grad = torch.autograd.grad(res_norm.sum(), x_flow)[0]

        with torch.no_grad():
            # measurement update: x_t_flow -> x_t_flow_guided
            step = zeta * grad
            if max_rel_step is not None:
                limit = max_rel_step * math.sqrt(d) * max(sigma_t_prime.item(), sigma_floor)
                step_norm = step.flatten(1).norm(dim=1).clamp_min(1e-12)
                step = step * (limit / step_norm).clamp(max=1.0).view(-1, 1, 1, 1)
            x_t_flow_guided = x_t_flow - step

            x_t = x_t_flow_guided
            t = t - t_delta

    return x_t_flow_guided




def stmd_inference_guided_dps(model, n_inf, n_mf, samples_shape, device, A, y, zeta_prime,):
    """
    DPS Alg. 1 applied to STMD. Per outer iteration:
      x_0_hat = mean-flow sample conditioned on x_t
      x_t_flow = bridge(x_0_hat, x_t)
      x_t_new = x_t_flow - zeta_i * grad_{x_t} ||y - A(x_0_hat)||^2
    with zeta_i = zeta_prime / ||y - A(x_0_hat)||
    """
    B = samples_shape[0]
    t = 1.0
    t_delta = 1 / n_inf
    s_delta = 1 / n_mf

    x_t = torch.randn(samples_shape, device=device)

    for k in range(n_inf):
        # mean-flow pipeline with autograd on
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

            # squared residual, DPS style
            res_sq = (y - A(x_0_hat)).flatten(1).pow(2).sum(dim=1)   # (B,)
            grad_sq = torch.autograd.grad(res_sq.sum(), x_t)[0]

        with torch.no_grad():
            x_0_hat, x_t = x_0_hat.detach(), x_t.detach()

            # unguided bridge: (x_t, x_0_hat) -> x_t_flow
            alpha_t, sigma_t = noise_schedule(t=torch.tensor([t], device=device))
            alpha_tp, sigma_tp = noise_schedule(t=torch.tensor([t - t_delta], device=device))

            mu = (
                (alpha_tp**2 - alpha_t**2) / (alpha_tp * sigma_t**2) * x_0_hat
                + (alpha_t * sigma_tp**2) / (alpha_tp * sigma_t**2) * x_t
            )
            var = (sigma_tp**2 * (alpha_tp**2 - alpha_t**2)) / (alpha_tp**2 * sigma_t**2)
            x_t_flow = mu + torch.sqrt(var) * torch.randn(samples_shape, device=device)

            # measurement update with zeta_i = zeta_prime / ||residual||
            zeta_i = zeta_prime / res_sq.sqrt().clamp_min(1e-12)
            x_t = x_t_flow - zeta_i.view(-1, 1, 1, 1) * grad_sq
            t = t - t_delta

    return x_t