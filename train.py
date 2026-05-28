import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.sr_dataset import SRTrainDataset, SRValDataset
from models.srresnet import SRResNet
from utils.io import ensure_dir, save_json
from utils.metrics import calc_psnr, calc_ssim


def validate(model, loader, device):
    model.eval()
    psnr_sum, ssim_sum = 0.0, 0.0
    with torch.no_grad():
        for lr, hr, _ in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_sum += calc_psnr(sr, hr)
            ssim_sum += calc_ssim(sr, hr)
    n = len(loader)
    return psnr_sum / n, ssim_sum / n


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.checkpoint_dir)

    train_set = SRTrainDataset(args.train_hr_dir, scale=args.scale, patch_size=args.patch_size, augment=True)
    val_set = SRValDataset(args.val_hr_dir, scale=args.scale)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = SRResNet(scale=args.scale, num_blocks=args.num_blocks).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_psnr = 0.0
    history = {"train_loss": [], "val_psnr": [], "val_ssim": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        val_psnr, val_ssim = validate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, PSNR={val_psnr:.3f}, SSIM={val_ssim:.4f}")

        ckpt_last = Path(args.checkpoint_dir) / "last.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch, "history": history}, ckpt_last)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            ckpt_best = Path(args.checkpoint_dir) / "best_psnr.pth"
            torch.save({"model": model.state_dict(), "epoch": epoch, "psnr": best_psnr}, ckpt_best)

    save_json(history, str(Path(args.checkpoint_dir) / "history.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hr_dir", type=str, required=True)
    parser.add_argument("--val_hr_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/srresnet_x4")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--num_blocks", type=int, default=16)
    args = parser.parse_args()
    main(args)
