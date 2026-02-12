
import json, sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

project_dir = os.environ["TEST_PROJECT_DIR"]
hyper_file = os.environ["TEST_HYPER_FILE"]
model_name = os.environ["TEST_MODEL_NAME"]

sys.path.insert(0, os.path.join(project_dir, "training_scripts"))

import torch
from torch_geometric.loader import DataLoader
from train_graph import (
    get_model_class, translate_model_config, adapt_model_config_from_data,
    make_optimizer, make_loss, MATBatchWrapper, load_dataset_pyg
)

with open(hyper_file) as f:
    hyper = json.load(f)

entry = hyper[model_name]
data_config = entry.get("data", {})

pyg_list = load_dataset_pyg(data_config)

model_config = dict(entry["model"]["config"])
model_config.pop("model_name", None)
model_config = translate_model_config(model_name, model_config)
model_config = adapt_model_config_from_data(model_name, model_config, pyg_list)

ModelClass = get_model_class(model_name)
model = ModelClass(**model_config)
if model_name == "MAT":
    model = MATBatchWrapper(model)
model.train()

training_config = entry.get("training", {})
compile_config = training_config.get("compile", {})
optimizer = make_optimizer(model, compile_config)
loss_fn = make_loss(compile_config)

output_embedding = entry["model"]["config"].get("output_embedding", "graph")
label_key = "node_labels" if output_embedding == "node" else "y"

subset = pyg_list[:min(8, len(pyg_list))]
loader = DataLoader(subset, batch_size=min(4, len(subset)))
batch = next(iter(loader))

optimizer.zero_grad()
pred = model(batch)
target = getattr(batch, label_key, batch.y)
if target.dim() == 1:
    target = target.unsqueeze(-1)
if pred.shape != target.shape:
    target = target[:pred.shape[0]]
    if target.dim() > 1 and pred.dim() > 1 and target.shape[1] != pred.shape[1]:
        target = target[:, :pred.shape[1]]
target = target.float()
loss = loss_fn(pred, target)
loss.backward()
optimizer.step()

optimizer.zero_grad()
pred2 = model(batch)
target2 = getattr(batch, label_key, batch.y)
if target2.dim() == 1:
    target2 = target2.unsqueeze(-1)
if pred2.shape != target2.shape:
    target2 = target2[:pred2.shape[0]]
    if target2.dim() > 1 and pred2.dim() > 1 and target2.shape[1] != pred2.shape[1]:
        target2 = target2[:, :pred2.shape[1]]
target2 = target2.float()
loss2 = loss_fn(pred2, target2)

from torch.nn.parameter import UninitializedParameter
n_params = sum(p.numel() for p in model.parameters()
               if not isinstance(p, UninitializedParameter))
print("OK loss=%.4f->%.4f shape=%s params=%d" % (
    loss.item(), loss2.item(), tuple(pred.shape), n_params))
