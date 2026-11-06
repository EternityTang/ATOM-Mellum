"""Reserved Mellum entry into ATOM's existing MTP proposer/verifier framework.

The released Mellum checkpoint has no prediction head. This entry deliberately
refuses activation instead of guessing a DeepSeek/MiMo predictor architecture
or running uninitialized draft weights. The target model does not import it.
"""

import json
import re
from pathlib import Path

from torch import nn


class MellumMTPUnavailable(RuntimeError):
    pass


def inspect_mellum_mtp(model_path):
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    index_path = path / "model.safetensors.index.json"
    names = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else {}
    base_layers = config["num_hidden_layers"]
    head_keys = []
    for name in names:
        layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if (
            re.search(r"(?:^|\.)(?:mtp|nextn|eh_proj|enorm|hnorm)(?:\.|$)", name)
            or layer and int(layer.group(1)) >= base_layers
        ):
            head_keys.append(name)
    return {
        "checkpoint": str(path),
        "checkpoint_tensors": len(names),
        "head_tensor_count": len(head_keys),
        "head_tensor_examples": head_keys[:8],
        "enabled": False,
        "ready": False,
        "framework": "ATOM EagleProposer and existing target verification/rejection sampling",
        "prediction_head_status": "missing" if not head_keys else "layout_not_yet_implemented",
    }


def prepare_mellum_mtp_config(config, model_path):
    report = inspect_mellum_mtp(model_path)
    if not report["head_tensor_count"]:
        raise MellumMTPUnavailable(
            "Mellum MTP is disabled: this checkpoint has no MTP head weights. "
            "Keep --method unset; official Mellum head weights and architecture "
            "are required before enabling ATOM's DeepSeek/MiMo-style MTP framework."
        )
    raise MellumMTPUnavailable(
        "Mellum MTP tensors were found, but their predictor layout is not yet "
        "implemented or validated. Keep MTP disabled; do not substitute a "
        "DeepSeek/MiMo model's weights or assume its architecture."
    )


class MellumMTP(nn.Module):
    """Lazy registry entry; not a runnable predictor until a head is supported."""

    def __init__(self, atom_config, prefix=""):
        super().__init__()
        prepare_mellum_mtp_config(atom_config.hf_config, atom_config.model)
