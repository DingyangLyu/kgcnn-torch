"""End-to-end training test: ESOL + GCN (molecule regression)."""
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from kgcnn_torch.data.datasets.ESOLDataset import ESOLDataset
from kgcnn_torch.models.gcn import GCNModel

# Load dataset
print("Loading ESOL dataset...")
t0 = time.time()
dataset = ESOLDataset(root="/tmp/kgcnn_train_esol")
print(f"  {len(dataset)} graphs in {time.time()-t0:.1f}s")
print(f"  Example: {dataset[0]}")

# Train/val split (use subset for speed)
torch.manual_seed(42)
n = min(500, len(dataset))
perm = torch.randperm(len(dataset))[:n]
subset = dataset[perm]
n_train = int(0.8 * n)
train_ds = subset[:n_train]
val_ds = subset[n_train:]
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model - GCN with embedding (uses z -> embedding)
model = GCNModel(
    node_dim=64,
    depth=3,
    gcn_units=64,
    num_targets=1,
    output_embedding="graph",
    output_final_activation="linear",
    output_units=[32, 16],
    use_node_embedding=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"  Model: {sum(p.numel() for p in model.parameters())} params")

# Training loop
print("\nTraining...")
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(batch).squeeze(-1)
        loss = F.mse_loss(pred, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    train_mse = total_loss / n_batches

    # Validation
    model.eval()
    val_loss = 0
    n_val = 0
    with torch.no_grad():
        for batch in val_loader:
            pred = model(batch).squeeze(-1)
            val_loss += F.mse_loss(pred, batch.y.float()).item()
            n_val += 1
    val_mse = val_loss / max(n_val, 1)

    if epoch % 2 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: train_MSE={train_mse:.4f}, val_MSE={val_mse:.4f}")

print(f"\nFinal: train_MSE={train_mse:.4f}, val_MSE={val_mse:.4f}")
print("ESOL training test PASSED")
