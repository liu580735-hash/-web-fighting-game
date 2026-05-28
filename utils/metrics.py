import numpy as np
import torch
from skimage.metrics import structural_similarity


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(0, 1)
    img = t.squeeze(0).permute(1, 2, 0).numpy()
    return img


def calc_psnr(sr: torch.Tensor, hr: torch.Tensor, eps=1e-10) -> float:
    mse = torch.mean((sr - hr) ** 2).item()
    return 10.0 * np.log10(1.0 / (mse + eps))


def calc_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    sr_img = tensor_to_image(sr)
    hr_img = tensor_to_image(hr)
    return structural_similarity(sr_img, hr_img, data_range=1.0, channel_axis=2)
