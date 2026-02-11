"""End-to-end training test: MatProjectEForm + SchNet (crystal property prediction)."""
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from kgcnn_torch.models.schnet import SchNetModel

print("Loading MatProjectEForm dataset...")
t0 = time.time()
from kgcnn_torch.data.datasets.MatProjectEFormDataset import MatProjectEFormDataset
dataset = MatProjectEFormDataset(root="/tmp/kgcnn_train_matproj")
print(f"  {len(dataset)} graphs in {time.time()-t0:.1f}s")

d = dataset[0]
print(f"  Example: {d}")
has_ei = hasattr(d, 'edge_index') and d.edge_index is not None
print(f"  Has edge_index: {has_ei}")

if not has_ei:
    print("  Adding RadiusGraph transform...")
    from torch_geometric.transforms import RadiusGraph
    dataset = MatProjectEFormDataset(root="/tmp/kgcnn_train_matproj",
                                     transform=RadiusGraph(r=5.0, max_num_neighbors=32))
    d = dataset[0]
    print(f"  After transform: {d}")

# Small subset
torch.manual_seed(42)
n = min(500, len(dataset))
perm = torch.randperm(len(dataset))[:n]
subset = dataset[perm]
n_train = int(0.8 * n)
train_ds = subset[:n_train]
val_ds = subset[n_train:]
print(f"  Using {n} graphs — Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# SchNet
model = SchNetModel(
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"  Model: {sum(p.numel() for p in model.parameters())} params")

print("\nTraining...")
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)
        target = batch.y.float().squeeze(-1)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    train_mse = total_loss / n_batches

    if epoch % 2 == 0 or epoch == 1:
        model.eval()
        val_loss = n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch).squeeze(-1)
                target = batch.y.float().squeeze(-1)
                val_loss += F.mse_loss(pred, target).item()
                n_val += 1
        print(f"  Epoch {epoch:3d}: train_MSE={train_mse:.4f}, val_MSE={val_loss/n_val:.4f}")

print("\nMatProject training test PASSED")
