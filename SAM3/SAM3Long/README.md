# SAM3Long

This folder contains the SAM3Long backend and VOT/VOS-style tools. The recommended way to run it with the other SAM3 trackers is through the root unified runner.

## Recommended Usage

From the repository root:

```bash
python run_all_models.py   --config config.local.yaml   --models SAM3Long   --dataset lasot
```

With an explicit checkpoint:

```bash
python run_all_models.py   --config config.local.yaml   --models SAM3Long   --dataset lasot   --sam3-checkpoint checkpoints/sam3.pt
```

Visualize predictions:

```bash
python run_all_models.py   --config config.local.yaml   --models SAM3Long   --dataset didi   --visualize
```

## Config

SAM3Long reads root `config.local.yaml` when invoked through `run_all_models.py`. The most relevant fields are:

```yaml
models:
  sam3:
    checkpoint: /path/to/sam3_checkpoint.pt
    load_from_hf: true
    compile: false
    apply_temporal_disambiguation: true
```

`tools/vot_config.yaml` is kept as a public template for direct tool usage and does not contain private dataset paths.

## Direct Tool Usage

```bash
python SAM3Long/tools/vot_inference.py   --config_file config.local.yaml   --dataset_name lasot   --dataset_dir /path/to/LaSOT   --output_dir outputs/sam3long_lasot
```

Use `--disable_temporal_disambiguation` only when you intentionally want to disable SAM3 temporal memory selection. By default, the value follows the config file.

## Files

- `tools/vot_inference.py` is the main VOT-style inference tool.
- `tools/vos_inference.py` contains VOS-style inference utilities.
- `tools/vot_config.yaml` is a public template config.
- `didi-workspace/` contains a small VOT toolkit workspace template.

## Notes

This backend uses the shared root `utils/` package where applicable. CUDA is expected for normal inference.
