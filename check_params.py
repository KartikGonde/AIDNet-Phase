import torch
from model import Network

model = Network()
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params: {total:,}")

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = model(x)
print(f"Input shape:  {tuple(x.shape)}")
print(f"Output shape: {tuple(out.shape)}")

print("\nPer-module breakdown:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if params > 0:
        print(f"  {name:20s} {params:>10,}")
