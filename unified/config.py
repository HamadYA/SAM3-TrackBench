from pathlib import Path
import os

import yaml


DATASET_CONFIG_KEYS = {
    "didi": "didi_dataset_path",
    "got_10k": "got_10k_dataset_path",
    "lasot": "lasot_dataset_path",
    "lasot_ext": "lasot_ext_dataset_path",
    "trackingnet": "trackingnet_dataset_path",
    "tnl2k": "tnl2k_dataset_path",
    "latot": "latot_dataset_path",
    "otb": "otb_dataset_path",
}


def find_repo_root(start=None):
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "run_all_models.py").exists() and (candidate / "config.yaml").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def config_path(local_file=None):
    env_config = os.environ.get("SAM3_CONFIG")
    if env_config:
        return Path(env_config).expanduser().resolve()

    repo_root = find_repo_root(local_file)
    local_config = repo_root / "config.local.yaml"
    if local_config.exists():
        return local_config
    return repo_root / "config.yaml"


def load_config(local_file=None):
    path = config_path(local_file)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def dataset_path(config, dataset_name):
    legacy_key = DATASET_CONFIG_KEYS.get(dataset_name)
    if legacy_key and config.get(legacy_key):
        return config[legacy_key]

    dataset_config = (config.get("datasets") or {}).get(dataset_name)
    if isinstance(dataset_config, dict):
        return dataset_config.get("path") or dataset_config.get("dataset_dir")
    return dataset_config
