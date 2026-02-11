"""End-to-end training test: MD17Revised + SchNet + EnergyForceModel (force prediction)."""
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph
from kgcnn_torch.models.schnet import SchNetModel
from kgcnn_torch.models.force import EnergyForceModel

# Link pre-downloaded tar
root = "/tmp/kgcnn_train_force"
raw_dir = os.path.join(root, "raw")
tar_src = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch/data/rmd17.tar.bz2"
os.makedirs(raw_dir, exist_ok=True)
tar_dst = os.path.join(raw_dir, "rmd17.tar.bz2")
if not os.path.exists(tar_dst) and os.path.exists(tar_src):
    os.symlink(tar_src, tar_dst)

# RadiusGraph transform to add edges
transform = RadiusGraph(r=5.0, max_num_neighbors=32)

print("Loading MD17Revised (ethanol)...")
t0 = time.time()
from kgcnn_torch.data.datasets.MD17RevisedDataset import MD17RevisedDataset
dataset = MD17RevisedDataset(trajectory_name="ethanol", root=root, transform=transform)
print(f"  {len(dataset)} graphs in {time.time()-t0:.1f}s")
print(f"  Example: {dataset[0]}")

# Small subset (rMD17 recommends <= 1000)
torch.manual_seed(42)
perm = torch.randperm(len(dataset))[:200]
subset = dataset[perm]
n_train = 160
train_ds = subset[:n_train]
val_ds = subset[n_train:]
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# SchNet energy model -> EnergyForceModel wrapper
energy_model = SchNetModel(
    node_dim=32,
    depth=2,
    units=32,
    edge_dim=20,
    gauss_bins=20,
    gauss_distance=5.0,
    num_targets=1,
    output_embedding="graph",
    make_distance=True,
    expand_distance=True,
)

model = EnergyForceModel(
    energy_model=energy_model,
    coordinate_input="pos",
    output_as_dict=True,
    is_physical_force=True,
    output_squeeze_states=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"  Model: {sum(p.numel() for p in model.parameters())} params")

# Training
print("\nTraining (energy + force loss)...")
force_weight = 10.0

for epoch in range(1, 11):
    model.train()
    total_e_loss = total_f_loss = 0
    n_batches = 0

    for batch in train_loader:
        batch.pos = batch.pos.detach().requires_grad_(True)
        optimizer.zero_grad()
        out = model(batch)
        e_loss = F.mse_loss(out["energy"].squeeze(-1), batch.energy.squeeze(-1))
        f_loss = F.mse_loss(out["force"], batch.force)
        loss = e_loss + force_weight * f_loss
        loss.backward()
        optimizer.step()
        total_e_loss += e_loss.item()
        total_f_loss += f_loss.item()
        n_batches += 1

    if epoch % 2 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: E_MSE={total_e_loss/n_batches:.6f}, F_MSE={total_f_loss/n_batches:.6f}")

# Quick validation
with torch.enable_grad():
    model.eval()
    ve = vf = 0
    nv = 0
    for batch in val_loader:
        batch.pos = batch.pos.detach().requires_grad_(True)
        out = model(batch)
        ve += F.mse_loss(out["energy"].squeeze(-1), batch.energy.squeeze(-1)).item()
        vf += F.mse_loss(out["force"], batch.force).item()
        nv += 1
print(f"\nVal: E_MSE={ve/nv:.6f}, F_MSE={vf/nv:.6f}")
print("Force training test PASSED")
