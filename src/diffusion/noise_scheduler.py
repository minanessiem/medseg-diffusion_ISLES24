import math
import numpy as np
import torch

class NoiseScheduler:
    def __init__(self, timesteps, beta_min=0.0001, beta_max=0.02, mode='linear', cosine_s=8e-3):
        """
        Args:
            timesteps (int): The number of timesteps for the noise schedule.
            beta_min (float): The minimum value of beta in the schedule.
            beta_max (float): The maximum value of beta in the schedule.
            mode (str): The type of scheduling to use. Options are 'quad', 'linear', 'cosine'.
            cosine_s (float): S parameter for cosine scheduling. Defaults to 8e-3.
        """
        self.timesteps = timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.mode = mode
        self.cosine_s = cosine_s
        assert mode in ['quad', 'linear', 'cosine'], 'Unsupported scheduling mode'

    def get_beta_schedule(self):
        if self.mode == "quad":
            betas = (
                torch.linspace(
                    self.beta_min ** 0.5, self.beta_max ** 0.5, self.timesteps, dtype=torch.float64
                )
                ** 2
            )
        elif self.mode == "linear":
            betas = torch.linspace(
                self.beta_min, self.beta_max, self.timesteps, dtype=torch.float64
            )
        elif self.mode == "cosine":
            timesteps = (
                torch.arange(self.timesteps + 1, dtype=torch.float64) / self.timesteps + self.cosine_s
            )
            alphas = timesteps / (1 + self.cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(min=0 ,max=0.999)

        return betas.numpy()

    def get_alpha_schedule(self):
        beta_schedule = self.get_beta_schedule()
        alpha_schedule = 1 - beta_schedule
        return alpha_schedule

    def get_alpha_bar_schedule(self):
        alpha_schedule = self.get_alpha_schedule()
        alpha_bar_schedule = np.cumprod(alpha_schedule)
        return alpha_bar_schedule

    def get_alphas_bar_previous_schedule(self):
        alpha_bar_schedule = self.get_alpha_bar_schedule()
        alpha_bars_prev = np.pad(alpha_bar_schedule[:-1], (1, 0), mode='constant', constant_values=1)
        return alpha_bars_prev
