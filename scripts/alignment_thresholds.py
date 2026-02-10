#!/usr/bin/env python3
"""Central threshold configuration for layerwise and model-level alignment assertions."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class Threshold:
    max_mae: float
    max_abs: float


# Layerwise defaults (tight: single-layer comparison)
DEFAULT_THRESHOLD = Threshold(max_mae=1e-6, max_abs=1e-5)

# Model-level defaults (looser: errors accumulate through full model)
MODEL_DEFAULT_THRESHOLD = Threshold(max_mae=1e-5, max_abs=1e-4)

# Training alignment defaults (tight: same torch autograd engine on both sides)
TRAINING_DEFAULT_THRESHOLD = Threshold(max_mae=1e-6, max_abs=1e-5)

THRESHOLDS_BY_SCRIPT: Dict[str, Threshold] = {
    # Layerwise overrides
    "align_hdnnp2nd_layerwise.py": Threshold(max_mae=2e-5, max_abs=1e-3),
    "align_hdnnp2nd_behler_layerwise.py": Threshold(max_mae=1e-5, max_abs=1e-4),
    # Model-level overrides for numerically challenging models
    "align_cgcnn_model.py": Threshold(max_mae=1e-4, max_abs=1e-3),
    "align_dimenetpp_model.py": Threshold(max_mae=1e-4, max_abs=1e-3),
    "align_hdnnp2nd_model.py": Threshold(max_mae=1e-4, max_abs=1e-3),
    "align_mxmnet_model.py": Threshold(max_mae=1e-4, max_abs=1e-3),
    "align_painn_model.py": Threshold(max_mae=1e-5, max_abs=1e-3),
}


def get_thresholds(script_path: str) -> Tuple[float, float]:
    script_name = Path(script_path).name
    threshold = THRESHOLDS_BY_SCRIPT.get(script_name)
    if threshold is None:
        # Use model-level default for *_model.py scripts, layerwise default otherwise
        if script_name.endswith("_model.py"):
            threshold = MODEL_DEFAULT_THRESHOLD
        else:
            threshold = DEFAULT_THRESHOLD
    return threshold.max_mae, threshold.max_abs
