import numpy as np
import os
import sys
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.loader import get_validation_data
import utils
from model import Network
from skimage import img_as_ubyte


parser = argparse.ArgumentParser(description="Test AIDNet-Phase")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["RICE", "Sate1K_Thin", "Sate1K_Moderate", "Sate1K_Thick"])
parser.add_argument("--gpus", default="0", type=str)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (auto-resolved from dataset if not given)")
parser.add_argument("--save_images", action="store_true", default=False)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

dataset_map = {
    "RICE":           ("./testing_data/RICE/",            "./results/Dehazed/RICE",        "./checkpoints/RICE/RICE.pth"),
    "Sate1K_Thin":    ("./testing_data/Sate1K/Thin/",    "./results/Dehazed/Sate1K_Thin", "./checkpoints/Sate1K_Thin/Sate1K_Thin_best.pth"),
    "Sate1K_Moderate":("./testing_data/Sate1K/Moderate/", "./results/Dehazed/Sate1K_Moderate", "./checkpoints/Sate1K_Moderate/Sate1K_Moderate_best.pth"),
    "Sate1K_Thick":   ("./testing_data/Sate1K/Thick/",   "./results/Dehazed/Sate1K_Thick", "./checkpoints/Sate1K_Thick/Sate1K_Thick_best.pth"),
}

input_dir, result_dir, default_ckpt = dataset_map[args.dataset]
checkpoint_dir = args.checkpoint or default_ckpt

utils.mkdir(result_dir)

test_dataset = get_validation_data(input_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

# Load model
model = Network()
checkpoint = torch.load(checkpoint_dir, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.cuda()
model.eval()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model params: {total_params / 1e6:.2f} M")
print(f"Testing on {args.dataset} dataset ({len(test_dataset)} images)")
print(f"Checkpoint: {checkpoint_dir}")


def expand2square(timg, factor=128.0):
    _, _, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(factor)) * factor)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)
    img[:, :, (X - h) // 2:(X - h) // 2 + h, (X - w) // 2:(X - w) // 2 + w] = timg
    mask[:, :, (X - h) // 2:(X - h) // 2 + h, (X - w) // 2:(X - w) // 2 + w].fill_(1.0)
    return img, mask


with torch.no_grad():
    psnr_vals = []
    ssim_vals = []

    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))

        rgb_noisy, mask = expand2square(data_test[1].cuda(), factor=128)
        filenames = data_test[2]

        rgb_restored = model(rgb_noisy)
        rgb_restored = torch.masked_select(rgb_restored, mask.bool()).reshape(
            1, 3, rgb_gt.shape[0], rgb_gt.shape[1]
        )
        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        psnr_val = utils.myPSNR(
            torch.from_numpy(rgb_gt).permute(2, 0, 1).unsqueeze(0),
            torch.from_numpy(rgb_restored).permute(2, 0, 1).unsqueeze(0),
        ).item()
        psnr_vals.append(psnr_val)

        if args.save_images:
            utils.save_img(
                os.path.join(result_dir, filenames[0]),
                img_as_ubyte(rgb_restored),
            )

    avg_psnr = sum(psnr_vals) / len(psnr_vals) if psnr_vals else 0.0
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
