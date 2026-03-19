# AIDNet-Phase — Implementation Guide

This project implements **AIDNet-Phase**, a lightweight phase-attention variant of AIDNet for aerial image dehazing. Build it by modifying the AIDNet codebase and borrowing modules from the Phaseformer codebase.

**Target: ~2.38M parameters** (down from AIDNet's 20.32M)

## Source Codebases

| Codebase | Path | Role |
|---|---|---|
| **AIDNet** | `../AIDTransformer-main/` | Base project structure — reuse `train.py`, `test.py`, `dataset.py`, `utils/` |
| **Phaseformer** | `../Phaseformer-main/` | Source for lightweight modules to copy into the new model |

## What to Copy from Each Codebase

### From `../Phaseformer-main/model_with_eca.py` — copy these classes/functions directly:

| Source Class/Function | What It Is | Use As |
|---|---|---|
| `inv_mag(x)` (line 13) | Phase Extraction Module (PEM). Uses `torch.fft.fft2` → discard amplitude → `ifft2`. **Zero learnable parameters.** | PEM for generating phase-based Q and K |
| `MDTA` (line 62) | Phase-based Multi-head Transposed Attention. Calls `inv_mag()` on Q and K after projection. Uses `self.temperature` as learnable scalar. | Replace AIDNet's `WindowAttention` + `ConvProjection` + `SepConv2d` |
| `GDFN` (line 90) | Gated Depthwise Feed-Forward Network with 2.66× expansion. Conv1×1 → DWConv3×3 → gated multiply → Conv1×1. | Replace AIDNet's `LeFF` / `Mlp` |
| `TransformerBlock` (line 125) | Wraps `MDTA` + `GDFN` + `LayerNorm`. Works in `(B, C, H, W)` format. | Replace AIDNet's `Deformable_Attentive_Transformer` |
| `ECA` (line 18) | Optimized Phase Attention Block (OPAB). Calls `inv_mag()` → GAP → adaptive 1D conv → sigmoid → channel reweighting. | Replace AIDNet's edge boosting skip connections (difference features) |
| `DownSample` (line 143) | Conv3×3 (C → C//2) + PixelUnshuffle(2) → output is 2C channels at half resolution. | Replace AIDNet's `Downsample` (Conv4×4, stride=2) |
| `UpSample` (line 153) | Conv3×3 (C → 2C) + PixelShuffle(2) → output is C//2 channels at double resolution. | Replace AIDNet's `Upsample` (ConvTranspose2d) |

### From `../AIDTransformer-main/` — reuse the training infrastructure:

| File | What to Reuse |
|---|---|
| `train.py` | Training loop, loss functions (`VGGPerceptualLoss`, edge loss), optimizer setup, data loading. Modify to import new model instead of `from model import Network`. |
| `test.py` | Testing/evaluation pipeline. Modify import. |
| `dataset.py` | Dataset class — reuse as-is. |
| `utils/` | All utility functions (`loader.py`, `image_utils.py`, `dir_utils.py`, `dataset_utils.py`) — reuse as-is. |

### From `../AIDTransformer-main/model.py` — DELETE these (do NOT copy):

| Class | Why Remove |
|---|---|
| `SepConv2d` | Contains `DeformConv2d` + `SpatialAttention` for offset generation — this is the heaviest component (~most of 20.32M params) |
| `ConvProjection` | Wraps `SepConv2d` for Q/K/V — replaced by `MDTA`'s internal Conv1×1+DWConv3×3 |
| `WindowAttention` | Window-based spatial attention — replaced by `MDTA`'s transposed channel attention |
| `Deformable_Attentive_Transformer` | Full transformer block with deformable attention — replaced by Phaseformer's `TransformerBlock` |
| `TRANSFORMER_BLOCK` | Wrapper for multiple `Deformable_Attentive_Transformer` — rebuild with `TransformerBlock` |
| Half-scale streams (`hs1`–`hs4`) and quarter-scale streams (`qs1`–`qs4`) | These add many extra transformer blocks for multi-scale edge boosting — replaced by lightweight `ECA` skip connections |

## Critical Format Difference

AIDNet's `model.py` uses **token format** `(B, L, C)` where `L = H*W` (flattened spatial). Phaseformer's `model_with_eca.py` uses **image format** `(B, C, H, W)`.

**Decision: Use Phaseformer's `(B, C, H, W)` format throughout the new model.** This means:
- Do NOT copy AIDNet's `InputProj` / `OutputProj` (they flatten to `(B, L, C)`)
- Use simple `nn.Conv2d(3, C, 3, padding=1, bias=False)` for input projection
- Use simple `nn.Conv2d(C, 3, 3, padding=1, bias=False)` for output projection
- `DownSample` / `UpSample` from Phaseformer already work in `(B, C, H, W)` format

## New Model Architecture

Build a new `model.py` containing a `Network` class (same name for easy swap in `train.py`).

### Architecture Config

```python
num_blocks = [2, 2, 2, 2, 2]  # enc1, enc2, bottleneck, dec2, dec1
num_heads  = [1, 2, 4, 2, 1]  # heads at each level
channels   = [48, 96, 192]     # 3-level hierarchy
expansion_factor = 2.66        # for GDFN (same as Phaseformer)
```

### Network Flow

```
Input (B, 3, H, W)
 │
 ├─ input_conv: nn.Conv2d(3, 48, 3, padding=1, bias=False)
 │
 ├─ encoder_l1: nn.Sequential(*[TransformerBlock(48, 1, 2.66) for _ in range(2)])
 │     │
 │     ├── eca_skip_1 = ECA(48)  ────────────────────────────┐
 │     │                                                      │
 │     ├─ down1: DownSample(48)  → (B, 96, H/2, W/2)         │
 │     │                                                      │
 │     ├─ encoder_l2: nn.Sequential(*[TransformerBlock(96, 2, 2.66) for _ in range(2)])
 │     │     │                                                │
 │     │     ├── eca_skip_2 = ECA(96) ────────────────┐       │
 │     │     │                                         │       │
 │     │     ├─ down2: DownSample(96) → (B, 192, H/4, W/4)    │
 │     │     │                                         │       │
 │     │     ├─ bottleneck: nn.Sequential(             │       │
 │     │     │    *[TransformerBlock(192, 4, 2.66)      │       │
 │     │     │      for _ in range(4)])                 │       │
 │     │     │                                         │       │
 │     │     ├─ up1: UpSample(192) → (B, 96, H/2, W/2)│       │
 │     │     │                                         │       │
 │     │     ├─ reduce2: nn.Conv2d(192, 96, 1, bias=F) │       │
 │     │     │  input = cat([up1_out, eca_skip_2_out]) ◄┘       │
 │     │     │                                                 │
 │     │     ├─ decoder_l2: nn.Sequential(                     │
 │     │           *[TransformerBlock(96, 2, 2.66)              │
 │     │             for _ in range(2)])                        │
 │     │                                                       │
 │     ├─ up2: UpSample(96) → (B, 48, H, W)                   │
 │     │                                                       │
 │     ├─ reduce1: nn.Conv2d(96, 48, 1, bias=False)            │
 │     │  input = cat([up2_out, eca_skip_1_out])          ◄────┘
 │     │
 │     ├─ decoder_l1: nn.Sequential(
 │           *[TransformerBlock(48, 1, 2.66) for _ in range(2)])
 │
 ├─ output_conv: nn.Conv2d(48, 3, 3, padding=1, bias=False) → residual
 │
 └─ output = input + residual
```

### Forward Method Pseudocode

```python
def forward(self, inp):
    x = self.input_conv(inp)

    # Encoder
    enc1 = self.encoder_l1(x)
    skip1 = self.eca_skip_1(enc1)

    enc2 = self.encoder_l2(self.down1(enc1))
    skip2 = self.eca_skip_2(enc2)

    # Bottleneck
    bot = self.bottleneck(self.down2(enc2))

    # Decoder
    dec2 = self.decoder_l2(self.reduce2(torch.cat([self.up1(bot), skip2], dim=1)))
    dec1 = self.decoder_l1(self.reduce1(torch.cat([self.up2(dec2), skip1], dim=1)))

    # Residual output
    return inp + self.output_conv(dec1)
```

## Loss Function

Reuse AIDNet's `train.py` loss setup. The key losses are:

```python
L_total = L1 + 5 * L_edge + 10 * L_perceptual
```

- `VGGPerceptualLoss` is already in `../AIDTransformer-main/train.py`
- Edge loss (Sobel-based) is already in `../AIDTransformer-main/train.py`
- Optionally add adaptive loss weighting from Phaseformer (learnable `nn.Parameter` weights)

## Training

Modify `../AIDTransformer-main/train.py`:
1. Change `from model import Network` to import your new model
2. Keep everything else (optimizer, scheduler, data loading, loss functions) the same
3. No other changes needed — the new `Network` class has the same `forward(x) → output` interface

## Parameter Verification

After building the model, verify:

```python
model = Network()
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params: {total:,}")  # expect ~2.38M
```

Compare: AIDNet original = 20.32M, Phaseformer = 1.77M, this variant target ≈ 2.38M

## Key Implementation Notes

1. All convolutions: `bias=False`
2. Phaseformer's `MDTA` already uses transposed attention (C×C map, not HW×HW) — efficient
3. `inv_mag()` is the PEM — zero learnable params, just FFT → discard amplitude → IFFT
4. `ECA` computes adaptive kernel size as `k = |log2(C)/gamma + b/gamma|_odd` with `gamma=2, b=1`
5. `GDFN` uses gated mechanism: `project_in` splits into two paths, one goes through GELU, then element-wise multiply, then `project_out`
6. Keep `expansion_factor=2.66` for GDFN (matches Phaseformer and Restormer)

## Source Papers

- **AIDNet**: "Aerial Image Dehazing with Attentive Deformable Transformers" — Kulkarni & Murala, CVPR 2023
- **Phaseformer**: "Phaseformer: Phase-based Attention Mechanism for Underwater Image Restoration and Beyond" — Khan et al., WACV 2024
