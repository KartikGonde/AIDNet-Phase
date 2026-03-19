@echo off
echo Starting training for dataset: Sate1K_Thin
echo Using L1 + 5xEdge + 10xPerceptual loss, batch_size=2 for 4GB VRAM
cd /d "%~dp0\.."
python train.py --dataset Sate1K_Thin --side_by_side --epochs 120 --batch_size 2 --patch_size 256 --num_workers 2
pause
