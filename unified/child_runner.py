import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml


DATASET_RUNNERS = {
    "didi": {
        "module": "run_didi",
        "kind": "didi",
        "config_key": "didi_dataset_path",
    },
    "got_10k": {
        "module": "run_got10k",
        "kind": "box",
        "config_key": "got_10k_dataset_path",
    },
    "lasot": {
        "module": "run_lasot",
        "kind": "box",
        "config_key": "lasot_dataset_path",
    },
    "lasot_ext": {
        "module": "run_lasot_ext",
        "kind": "box",
        "config_key": "lasot_ext_dataset_path",
    },
    "trackingnet": {
        "module": "run_trackingnet",
        "kind": "box",
        "config_key": "trackingnet_dataset_path",
    },
    "tnl2k": {
        "module": "run_tnl2k",
        "kind": "box",
        "config_key": "tnl2k_dataset_path",
    },
    "latot": {
        "module": "run_latot",
        "kind": "box",
        "config_key": "latot_dataset_path",
    },
    "otb": {
        "module": "run_otb",
        "kind": "box",
        "config_key": "otb_dataset_path",
    },
}


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def dataset_path_from_config(config, dataset_name):
    runner = DATASET_RUNNERS[dataset_name]
    legacy_path = config.get(runner["config_key"])
    if legacy_path:
        return legacy_path

    dataset_config = (config.get("datasets") or {}).get(dataset_name)
    if isinstance(dataset_config, dict):
        return dataset_config.get("path") or dataset_config.get("dataset_dir")
    return dataset_config


def import_model_runner(repo_root, model_dir, module_name):
    sys.path.insert(0, str(model_dir))
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    return importlib.import_module(module_name)


def run_standard_model(args):
    if args.dataset not in DATASET_RUNNERS:
        available = ", ".join(sorted(DATASET_RUNNERS))
        raise SystemExit(f"Unsupported dataset {args.dataset!r}. Available: {available}")

    repo_root = Path(args.repo_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    config_file = Path(args.config_file).resolve()

    os.environ["SAM3_CONFIG"] = str(config_file)
    os.environ["SAM3_MODEL_NAME"] = args.model_name
    os.chdir(repo_root)

    config = load_config(config_file)
    dataset_path = args.dataset_path or dataset_path_from_config(config, args.dataset)
    if not dataset_path:
        key = DATASET_RUNNERS[args.dataset]["config_key"]
        raise SystemExit(
            f"No path configured for dataset {args.dataset!r}. "
            f"Set {key!r} in {config_file}."
        )

    output_dir = None
    if not args.visualize:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    runner_info = DATASET_RUNNERS[args.dataset]
    module = import_model_runner(repo_root, model_dir, runner_info["module"])

    print(
        f"[{args.model_name}] dataset={args.dataset} "
        f"tracker={args.tracker_name} "
        f"{'visualize=True' if args.visualize else f'output={output_dir}'}",
        flush=True,
    )

    if runner_info["kind"] == "didi":
        if args.sequence:
            sequence_names = [args.sequence]
        else:
            sequence_names = module.get_seq_names(dataset_path)
        module.run_sequence(
            args.tracker_name,
            dataset_path,
            sequence_names,
            output_dir=None if args.visualize else str(output_dir),
        )
        return

    module.gt_path = dataset_path
    module.main(
        args.tracker_name,
        args.dataset,
        output_dir=None if args.visualize else str(output_dir),
        selected_sequence=args.sequence,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one standard SAM3 tracker backend in an isolated process."
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--sequence", default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_standard_model(args)


if __name__ == "__main__":
    main()
