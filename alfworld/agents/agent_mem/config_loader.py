"""Load ALFWorld YAML config without conflicting with script-level argparse."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml


def load_config_file(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    assert os.path.exists(config_path), f"Invalid config file: {config_path}"
    with open(config_path, encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    for param in overrides or []:
        fqn_key, value = param.split("=", 1)
        entry = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry = entry[k]
        entry[keys[-1]] = value
    return config


def get_max_steps_per_episode(config: Dict[str, Any], default: int = 50) -> int:
    """Match ALFWorld env/agent: max steps live under dagger.training or rl.training."""
    method = config.get("general", {}).get("training_method", "dagger")
    for key in (method, "dagger", "rl"):
        training = config.get(key, {}).get("training", {})
        if "max_nb_steps_per_episode" in training:
            return int(training["max_nb_steps_per_episode"])
    return int(config.get("env", {}).get("expert_timeout_steps", default))
