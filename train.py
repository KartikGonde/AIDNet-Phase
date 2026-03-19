import os
import sys
import math
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from model import Network
from utils.loader import get_training_data, get_validation_data
from utils.dataset_utils import MixUp_AUG
import utils


# ---------------------------------------------------------------------------
# Loss Functions (Paper Sec. 3.5)
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG-16 features from layers 3, 8, 15 (Paper Sec. 3.5)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])   # up to relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # up to relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)])  # up to relu3_3
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_f1 = self.slice1(pred)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)

        with torch.no_grad():
            tgt_f1 = self.slice1(target)
            tgt_f2 = self.slice2(tgt_f1)
            tgt_f3 = self.slice3(tgt_f2)

        loss = (F.l1_loss(pred_f1, tgt_f1)
                + F.l1_loss(pred_f2, tgt_f2)
                + F.l1_loss(pred_f3, tgt_f3))
        return loss


class EdgeLoss(nn.Module):
    """Edge loss using Laplacian convolution to extract high-frequency content."""

    def __init__(self):
        super().__init__()
        laplacian = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian_kernel", laplacian.repeat(3, 1, 1, 1))

    def _edges(self, x):
        return F.conv2d(x, self.laplacian_kernel, padding=1, groups=3)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand2square(timg, factor=128.0):
    """Pad image to the nearest multiple of *factor* (square)."""
    _, _, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(factor)) * factor)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)
    img[:, :, (X - h) // 2:(X - h) // 2 + h, (X - w) // 2:(X - w) // 2 + w] = timg
    mask[:, :, (X - h) // 2:(X - h) // 2 + h, (X - w) // 2:(X - w) // 2 + w].fill_(1.0)
    return img, mask


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    psnr_list = []
    for data in val_loader:
        target = data[0].to(device)
        inp = data[1].to(device)

        inp_padded, mask = expand2square(inp, factor=128)
        restored = model(inp_padded)
        restored = torch.masked_select(
            restored, mask.bool()
        ).reshape(1, 3, target.shape[2], target.shape[3])
        restored = torch.clamp(restored, 0, 1)

        psnr_list.append(utils.myPSNR(target, restored).item())

    model.train()
    return sum(psnr_list) / len(psnr_list) if psnr_list else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train AIDNet-Phase (Lightweight Aerial Dehazing)")

    # data
    p.add_argument("--dataset", type=str, required=True,
                   choices=["RICE", "Sate1K_Thin", "Sate1K_Moderate", "Sate1K_Thick"],
                   help="Dataset to train on")
    p.add_argument("--train_dir", type=str, default=None,
                   help="Override training data directory (must contain input/ and target/)")
    p.add_argument("--val_dir", type=str, default=None,
                   help="Override validation data directory")

    # training hyper-parameters (paper Sec. 4.2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=256,
                   help="Training patch size (M x M)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Initial learning rate")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixup", action="store_true", default=False,
                   help="Enable MixUp augmentation")

    # loss weights (paper Eq. 5)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_edge", type=float, default=5.0)
    p.add_argument("--lambda_perceptual", type=float, default=10.0)

    # misc
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--val_every", type=int, default=5,
                   help="Run validation every N epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args()


def get_data_dirs(args):
    """Resolve train/val directories from --dataset or explicit overrides."""
    dataset_map = {
        "RICE":           ("./training_data/RICE/",           "./testing_data/RICE/"),
        "Sate1K_Thin":    ("./training_data/Sate1K/Thin/",   "./testing_data/Sate1K/Thin/"),
        "Sate1K_Moderate":("./training_data/Sate1K/Moderate/","./testing_data/Sate1K/Moderate/"),
        "Sate1K_Thick":   ("./training_data/Sate1K/Thick/",  "./testing_data/Sate1K/Thick/"),
    }
    default_train, default_val = dataset_map[args.dataset]
    train_dir = args.train_dir or default_train
    val_dir = args.val_dir or default_val
    return train_dir, val_dir


def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- directories -----
    train_dir, val_dir = get_data_dirs(args)
    save_dir = os.path.join(args.save_dir, args.dataset)
    utils.mkdir(save_dir)

    print(f"Dataset     : {args.dataset}")
    print(f"Train dir   : {train_dir}")
    print(f"Val dir     : {val_dir}")
    print(f"Save dir    : {save_dir}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Patch size  : {args.patch_size}")
    print(f"LR          : {args.lr}")

    # ----- data -----
    train_dataset = get_training_data(train_dir, {"patch_size": args.patch_size})
    val_dataset = get_validation_data(val_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")

    # ----- model -----
    model = Network().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params / 1e6:.2f} M")

    # ----- loss -----
    criterion_l1 = nn.L1Loss().to(device)
    criterion_edge = EdgeLoss().to(device)
    criterion_perceptual = VGGPerceptualLoss().to(device)

    # ----- optimizer & scheduler (paper Sec. 4.2) -----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # ----- optional MixUp -----
    mixup = MixUp_AUG() if args.mixup else None

    # ----- resume -----
    start_epoch = 1
    best_psnr = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_psnr" in checkpoint:
            best_psnr = checkpoint["best_psnr"]
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed from epoch {start_epoch - 1}, best PSNR = {best_psnr:.2f}")

    # ----- training loop -----
    print("\n========== Training started ==========\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, data in enumerate(pbar):
            target = data[0].to(device)  # clean
            inp = data[1].to(device)     # hazy

            if mixup is not None and random.random() > 0.5:
                target, inp = mixup.aug(target, inp)

            restored = model(inp)
            restored = torch.clamp(restored, 0, 1)

            loss_l1 = criterion_l1(restored, target)
            loss_edge = criterion_edge(restored, target)
            loss_perceptual = criterion_perceptual(restored, target)

            loss = (args.lambda_l1 * loss_l1
                    + args.lambda_edge * loss_edge
                    + args.lambda_perceptual * loss_perceptual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_psnr = utils.batch_PSNR(restored, target, average=True).item()
            epoch_psnr += batch_psnr

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{batch_psnr:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Train PSNR: {avg_psnr:.2f} dB | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ----- validation -----
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_psnr = validate(model, val_loader, device)
            print(f"[Epoch {epoch}] Val PSNR: {val_psnr:.2f} dB")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {"epoch": epoch,
                     "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_psnr": best_psnr},
                    os.path.join(save_dir, f"{args.dataset}_best.pth"),
                )
                print(f"  -> New best model saved (PSNR: {best_psnr:.2f} dB)")

        # ----- periodic save -----
        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch,
                 "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_psnr": best_psnr},
                os.path.join(save_dir, f"{args.dataset}_epoch{epoch}.pth"),
            )

    # ----- final save -----
    torch.save(
        {"epoch": args.epochs,
         "state_dict": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(),
         "best_psnr": best_psnr},
        os.path.join(save_dir, f"{args.dataset}.pth"),
    )
    print(f"\nTraining complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"Final model saved to: {os.path.join(save_dir, args.dataset + '.pth')}")


if __name__ == "__main__":
    main()
