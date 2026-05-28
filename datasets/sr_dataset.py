import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0


def random_augment(hr: np.ndarray, lr: np.ndarray):
    if random.random() < 0.5:
        hr = np.fliplr(hr).copy()
        lr = np.fliplr(lr).copy()
    if random.random() < 0.5:
        hr = np.flipud(hr).copy()
        lr = np.flipud(lr).copy()
    if random.random() < 0.5:
        hr = np.rot90(hr).copy()
        lr = np.rot90(lr).copy()
    return hr, lr


class SRTrainDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int = 4, patch_size: int = 96, augment: bool = True):
        self.hr_paths = sorted(Path(hr_dir).glob("*"))
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        if len(self.hr_paths) == 0:
            raise ValueError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = _read_image(self.hr_paths[idx])

        h, w = hr.shape[:2]
        h = (h // self.scale) * self.scale
        w = (w // self.scale) * self.scale
        hr = hr[:h, :w]

        lr = cv2.resize(hr, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)

        lr_patch = self.patch_size // self.scale
        x = random.randint(0, lr.shape[1] - lr_patch)
        y = random.randint(0, lr.shape[0] - lr_patch)

        lr_crop = lr[y:y + lr_patch, x:x + lr_patch]
        hr_crop = hr[y * self.scale:(y + lr_patch) * self.scale, x * self.scale:(x + lr_patch) * self.scale]

        if self.augment:
            hr_crop, lr_crop = random_augment(hr_crop, lr_crop)

        return _to_tensor(lr_crop), _to_tensor(hr_crop)


class SRValDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int = 4):
        self.hr_paths = sorted(Path(hr_dir).glob("*"))
        self.scale = scale
        if len(self.hr_paths) == 0:
            raise ValueError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        path = self.hr_paths[idx]
        hr = _read_image(path)

        h, w = hr.shape[:2]
        h = (h // self.scale) * self.scale
        w = (w // self.scale) * self.scale
        hr = hr[:h, :w]

        lr = cv2.resize(hr, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)

        return _to_tensor(lr), _to_tensor(hr), path.name
