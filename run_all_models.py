import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = REPO_ROOT / "config.yaml"

DEFAULT_MODEL_ENTRIES = [
    {"name": "sam3", "path": "sam3", "type": "standard", "enabled": True},
    {"name": "SAMURAI", "path": "SAMURAI", "type": "standard", "enabled": True},
    {"name": "HiM2SAM", "path": "HiM2SAM", "type": "standard", "enabled": True},
    {"name": "DAM4SAM", "path": "DAM4SAM", "type": "standard", "enabled": True},
    {"name": "SAMITE", "path": "SAMITE", "type": "standard", "enabled": True},
    {"name": "SAM3Long", "path": "SAM3Long", "type": "sam3long", "enabled": True},
]

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


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def wrapper_config(config):
    return config.get("wrapper") or {}


def model_entries(config):
    entries = wrapper_config(config).get("models") or DEFAULT_MODEL_ENTRIES
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append(
                {"name": entry, "path": entry, "type": "standard", "enabled": True}
            )
        else:
            normalized.append(
                {
                    "name": entry["name"],
                    "path": entry.get("path", entry["name"]),
                    "type": entry.get("type", "standard"),
                    "enabled": entry.get("enabled", True),
                }
            )
    return normalized


def select_models(entries, requested_models):
    if not requested_models:
        return [entry for entry in entries if entry.get("enabled", True)]
    if requested_models == ["all"]:
        return entries

    by_name = {entry["name"]: entry for entry in entries}
    missing = [name for name in requested_models if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise SystemExit(
            f"Unknown model(s): {', '.join(missing)}. Available models: {available}"
        )
    return [by_name[name] for name in requested_models]


def dataset_path_from_config(config, dataset_name):
    legacy_key = DATASET_CONFIG_KEYS.get(dataset_name)
    if legacy_key and config.get(legacy_key):
        return config[legacy_key]

    dataset_config = (config.get("datasets") or {}).get(dataset_name)
    if isinstance(dataset_config, dict):
        return dataset_config.get("path") or dataset_config.get("dataset_dir")
    return dataset_config


def output_dir_for(output_root, model_name, dataset_name, tracker_name):
    return Path(output_root) / model_name / dataset_name / tracker_name


def resolve_config_path(path, config_file):
    if path is None or os.path.isabs(path):
        return path
    return str((Path(config_file).parent / path).resolve())


def sam3_checkpoint_from_config(config, config_file, model_name):
    model_config = (config.get("models") or {}).get(model_name or "sam3", {})
    checkpoint = (
        model_config.get("sam3_checkpoint")
        or model_config.get("checkpoint")
        or (config.get("weights") or {}).get("sam3")
    )
    return resolve_config_path(checkpoint, config_file)


def build_standard_command(args, config_file, entry, dataset_path, output_dir):
    command = [
        args.python,
        "-m",
        "unified.child_runner",
        "--repo-root",
        str(REPO_ROOT),
        "--config-file",
        str(config_file),
        "--model-name",
        entry["name"],
        "--model-dir",
        str((REPO_ROOT / entry["path"]).resolve()),
        "--dataset",
        args.dataset,
        "--tracker-name",
        args.tracker_name,
    ]
    if args.visualize:
        command.append("--visualize")
    else:
        command.extend(["--output-dir", str(output_dir)])
    if dataset_path:
        command.extend(["--dataset-path", dataset_path])
    if args.sequence:
        command.extend(["--sequence", args.sequence])
    return command


def build_sam3long_command(args, config_file, config, entry, dataset_path, output_dir):
    model_name = args.sam3_model_name or config.get("default_model") or "sam3"
    command = [
        args.python,
        str((REPO_ROOT / entry["path"] / "tools" / "vot_inference.py").resolve()),
        "--config_file",
        str(config_file),
        "--dataset_name",
        args.dataset,
        "--model_name",
        model_name,
    ]
    if args.visualize:
        command.append("--visualize")
    else:
        command.extend(["--output_dir", str(output_dir)])
    if dataset_path:
        command.extend(["--dataset_dir", dataset_path])
    if args.sam3_checkpoint:
        command.extend(["--sam3_checkpoint", args.sam3_checkpoint])
    else:
        checkpoint = sam3_checkpoint_from_config(config, config_file, model_name)
        if checkpoint:
            command.extend(["--sam3_checkpoint", checkpoint])
    if args.disable_hf_download:
        command.append("--disable_hf_download")
    if args.compile_model:
        command.append("--compile_model")
    if args.disable_temporal_disambiguation:
        command.append("--disable_temporal_disambiguation")
    if args.sequence:
        command.extend(["--sequence", args.sequence])
    return command


def build_command(args, config_file, config, entry, dataset_path, output_dir):
    model_type = entry.get("type", "standard")
    if model_type == "standard":
        return build_standard_command(args, config_file, entry, dataset_path, output_dir)
    if model_type == "sam3long":
        return build_sam3long_command(args, config_file, config, entry, dataset_path, output_dir)
    raise SystemExit(f"Unsupported model type {model_type!r} for {entry['name']!r}")


def run_command(command, env):
    print("\n$ " + shlex.join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run multiple SAM3 tracker folders from one wrapper."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_FILE),
        help="Shared wrapper config. Use config.local.yaml for private paths.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to run. Use 'all' to include disabled entries.",
    )
    parser.add_argument("--dataset", default=None, choices=sorted(DATASET_CONFIG_KEYS))
    parser.add_argument("--sequence", default=None, help="Optional single sequence name.")
    parser.add_argument("--tracker-name", default=None, help="Tracker variant name.")
    parser.add_argument(
        "--sam3-model-name",
        default=None,
        help="SAM3Long model entry in config.yaml. Defaults to config default_model.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        default=None,
        help="Optional local SAM3 checkpoint path for SAM3Long.",
    )
    parser.add_argument(
        "--disable-hf-download",
        action="store_true",
        help="Disable SAM3Long Hugging Face checkpoint download.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for SAM3Long.",
    )
    parser.add_argument(
        "--disable-temporal-disambiguation",
        action="store_true",
        help="Disable SAM3Long temporal disambiguation / memory selection.",
    )
    parser.add_argument("--output-root", default=None, help="Root output directory.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize predictions instead of writing output txt files.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first model command fails.",
    )
    parser.add_argument("--list-models", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config_file = Path(args.config).resolve()
    config = load_config(config_file)
    wrap_config = wrapper_config(config)

    entries = model_entries(config)
    if args.list_models:
        for entry in entries:
            status = "enabled" if entry.get("enabled", True) else "disabled"
            print(f"{entry['name']}\t{entry.get('type', 'standard')}\t{status}")
        return

    args.dataset = args.dataset or wrap_config.get("default_dataset") or "didi"
    args.tracker_name = args.tracker_name or wrap_config.get("default_tracker_name") or "sam3"
    output_root = args.output_root or wrap_config.get("output_root") or "outputs/unified"
    args.visualize = args.visualize or bool(wrap_config.get("visualize", False))
    fail_fast = args.fail_fast or bool(wrap_config.get("fail_fast", False))
    selected_entries = select_models(entries, args.models)

    dataset_path = dataset_path_from_config(config, args.dataset)
    if not dataset_path and not args.dry_run:
        key = DATASET_CONFIG_KEYS[args.dataset]
        raise SystemExit(
            f"No path configured for dataset {args.dataset!r}. Set {key!r} in {config_file}."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["SAM3_CONFIG"] = str(config_file)

    failures = []
    for entry in selected_entries:
        model_dir = REPO_ROOT / entry["path"]
        if not model_dir.exists():
            failures.append((entry["name"], f"missing directory: {model_dir}"))
            if fail_fast:
                break
            continue

        output_dir = output_dir_for(
            output_root,
            entry["name"],
            args.dataset,
            args.tracker_name,
        ).resolve()
        command = build_command(args, config_file, config, entry, dataset_path, output_dir)

        if args.dry_run:
            print(shlex.join(command))
            continue

        returncode = run_command(command, env)
        if returncode:
            failures.append((entry["name"], f"exit code {returncode}"))
            if fail_fast:
                break

    if failures:
        print("\nCompleted with failures:")
        for model_name, reason in failures:
            print(f"- {model_name}: {reason}")
        raise SystemExit(1)

    if args.dry_run:
        print("\nDry run complete.")
    else:
        print("\nAll selected model runs completed.")


if __name__ == "__main__":
    main()
