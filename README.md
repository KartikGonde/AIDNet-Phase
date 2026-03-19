# AIDNet-Phase: Lightweight Aerial Image Dehazing

**AIDNet-Phase** is a highly optimized, lightweight variant of [AIDNet](https://github.com/Karthik-Gonde/AIDTransformer-main) for aerial image dehazing. By substituting AIDNet's heavy attentive deformable transformers with phase-based attention modules inspired by [Phaseformer](https://github.com/Karthik-Gonde/Phaseformer-main), we've drastically reduced the model parameters from **20.32M** down to just **~2.38M** while preserving strong spatial and perceptual representation abilities.

---

## 🚀 Features

- **Phase Extraction Module (PEM):** Zero-parameter lightweight module that captures pure high-frequency spatial context using 2D FFT.
- **Phase-based Multi-head Transposed Attention (MDTA):** Replaces computationally heavy spatial attention with efficient transposed channel attention.
- **Gated Depthwise Feed-Forward Network (GDFN):** Introduces non-linear feature transformation via a gated depthwise convolution block.
- **Optimized Phase Attention Block (OPAB / ECA):** Streamlined phase-aware skip connections.
- **Minimal VRAM Requirements:** Fits perfectly for smaller GPUs (optimized for 4GB VRAM like the mobile RTX 3050).

---

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have **Python 3.10+** (Python 3.13 supported) installed.  
The core dependencies are tracked in `requirements.txt` and can be installed via:

```bash
pip install -r requirements.txt
```

### 2. PyTorch CUDA Installation (Crucial)
To ensure optimal performance, install the necessary PyTorch build matching your driver. 
For **CUDA 12.6/12.7**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

---

## 🗃️ Dataset Structure

### Standard Format (e.g., RICE)
Place classical paired datasets into separate `input/` (degraded/hazy) and `target/` (clean) directories:
```text
training_data/RICE/
    input/
        0001.png
    target/
        0001.png
```

### Sate1K Side-by-Side Format
The codebase has native data loaders (`DataLoaderTrainSideBySide`) that split horizontal side-by-side concatenated images automatically (haze on left, clean on right). The dataset should be stored as flat directories:

```text
training_data/Sate1K/
    Thin/
        0001.png
        0002.png
    Moderate/
        ...
    Thick/
        ...
```
*(Validation datasets follow the exact same structure inside the `testing_data/` directory)* 

---

## 🏃‍♂️ Training & Inference

We have generated explicit `.bat` scripts inside the `scripts/` directory configured specifically to prevent OOM errors on smaller GPUs (batch_size=2). 

To train the model on a specific condition, double-click the script or execute it via command line:
```bash
# Train on Sate1K Thin (Side-by-side)
scripts\train_sate1k_thin.bat

# Train on Sate1K Moderate (Side-by-side)
scripts\train_sate1k_moderate.bat

# Train on Sate1K Thick (Side-by-side)
scripts\train_sate1k_thick.bat

# Train on RICE standard (Separate input/target)
scripts\train_rice.bat
```

Checkpoints will routinely be exported to the dynamically mapped `checkpoints/` folder.

---

## ⚖️ Parameter Optimization

```
AIDNet Base:          20,320,000 parameters
AIDNet-Phase (Ours):   2,416,226 parameters (~11.9% of base size)
```
*Run `python check_params.py` to inspect the exact network parameter distribution.*
