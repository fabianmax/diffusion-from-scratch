import torch
import torch.nn.functional as F
from tqdm import tqdm


def extract(a, t, x_shape):
    """ Extract t index for a batch of indices.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def beta_schedule(timesteps, schedule_type: str = "linear"):
    """ Generates a beta schedule for the forward diffusion process of a DDPM.

    This schedule defines the variance of the noise added to the input at each timestep.
    Linear schedule as in https://arxiv.org/abs/2006.11239
    Cosine schedule as in https://arxiv.org/abs/2102.09672

    Args:
        timesteps: Number of timesteps
        schedule_type: Type of schedule. Either "linear" or "cosine"

    Returns:
        torch.Tensor: Beta schedule of shape [timesteps]
    """
    beta_start = 0.0001
    beta_end = 0.02
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule_type == "quadratic":
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    elif schedule_type == "sigmoid":
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif schedule_type == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError()


def create_diffusion_params(timesteps, beta_schedule_type: str = "linear"):
    """ Creates the parameters of the forward diffusion process of a DDPM.

    Args:
        timesteps: Number of total timesteps
        beta_schedule_type: Type of schedule. Either "linear" or "cosine"

    Returns:
        dict: Dictionary containing the parameters of the forward diffusion process
    """
    # define betas (variance)
    betas = beta_schedule(timesteps=timesteps, schedule_type=beta_schedule_type)

    # define alphas (mean)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {"timesteps": timesteps,
            "betas": betas,
            "alphas": alphas,
            "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
            "sqrt_recip_alphas": sqrt_recip_alphas,
            "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
            "posterior_variance": posterior_variance}


def q_sample(x_start, t, diff_dict, noise=None):
    """ Forward diffusion process of a DDPM.

    Given the real data distribution q(x_0) the forward diffusion process is defined as
    q(x_t | x_t-1) = N(x_t; sqrt(1- beta_t) * x_t-1, beta_t * I)

    Args:
        x_start: Image from the real data distribution q(x_0)
        t: Time index
        diff_dict: Dictionary containing the parameters of the forward diffusion process
        noise:  Noise added to the input at each timestep

    Returns:
        torch.Tensor: Image at time t incl noise
    """

    # extract parameters of diffusion process
    sqrt_alphas_cumprod = diff_dict["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = diff_dict["sqrt_one_minus_alphas_cumprod"]

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, diff_dict, noise=None, loss_type="l1"):
    """
    Loss function for DDPM as in https://arxiv.org/abs/2006.11239

    Objective function to learn the mean of the backward process using a neural network:
    q(x_t | x_t0) = N(x_t; sqrt(alpha_t * x_0), (1 - alpha_t) * I)

    Args:
        denoise_model: Model for noise prediction (U-Net)
        x_start: Image from the real data distribution q(x_0)
        t: Time step in diffusion process
        diff_dict: Dictionary containing the parameters of the forward diffusion process
        noise:  Noise added to the input at each timestep
        loss_type: Type of loss function. Either "l1", "l2" or "huber"
    Returns:
        torch.Tensor: Loss value
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # Create a noisy version of the input, then let the model predict the added noise based on the noisy input and
    # the current time step.
    x_noisy = q_sample(x_start=x_start, t=t, diff_dict=diff_dict, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    # Loss is the difference between the noise and the predicted noise
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index, diff_dict):
    """ Sampling algorithm to reverse the diffusion process using the mean function approximator µ_θ that learned to
     predict µ^hat_t

    See Algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf

    Args:
        model: Model for predict e (reparameterization of µ^hat_t)
        x: Image one diffusion step before (gets denoised)
        t: Time step in diffusion process
        t_index: Index of the current time step
        diff_dict: Dictionary containing the parameters of the forward diffusion process

    Returns:
        torch.Tensor: Image x_t-1 at time t-1 (denoised version of x_t)
    """
    # extract parameters of diffusion process
    betas = diff_dict["betas"]
    sqrt_one_minus_alphas_cumprod = diff_dict["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas = diff_dict["sqrt_recip_alphas"]
    posterior_variance = diff_dict["posterior_variance"]

    # extract time index of current time steps
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, diff_dict, shape):
    """ Sampling algorithm to reverse the diffusion process.

    Algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf

    Args:
        model: Model for predict e
        diff_dict: Dictionary containing the parameters of the forward diffusion process
        shape: Shape of the input image [batch_size, channels, height, width]

    Returns:
        torch.Tensor: Image x_t-1 at time t-1 (denoised version of x_t)
    """
    timesteps = diff_dict["timesteps"]
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model=model, x=img, t=t, t_index=i, diff_dict=diff_dict)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, diff_dict, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, diff_dict=diff_dict, shape=(batch_size, channels, image_size, image_size))
