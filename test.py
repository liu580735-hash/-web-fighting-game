import argparse
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from datasets.sr_dataset import SRValDataset
from models.srresnet import SRResNet
from utils.io import ensure_dir
from utils.metrics import calc_psnr, calc_ssim, tensor_to_image


def save_image(img, path):
    img = (img * 255.0).round().astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)

    dataset = SRValDataset(args.test_hr_dir, scale=args.scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SRResNet(scale=args.scale, num_blocks=args.num_blocks).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    psnr_sum, ssim_sum = 0.0, 0.0

    with torch.no_grad():
        for lr, hr, name in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)

            psnr_sum += calc_psnr(sr, hr)
            ssim_sum += calc_ssim(sr, hr)

            sr_img = tensor_to_image(sr)
            save_image(sr_img, Path(args.out_dir) / f"{Path(name[0]).stem}_SR.png")

    n = len(loader)
    print(f"Average PSNR: {psnr_sum / n:.3f} dB")
    print(f"Average SSIM: {ssim_sum / n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_hr_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/images")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=16)
    args = parser.parse_args()
    main(args)
