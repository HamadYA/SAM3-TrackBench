# DAM4SAM SAM3 Tracker

This folder contains the DAM4SAM-style SAM3 tracker backend. The recommended interface is the root unified runner, which handles shared config, output layout, and model selection.

## Recommended Usage

From the repository root:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models DAM4SAM \
  --dataset lasot
```

Run one sequence:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models DAM4SAM \
  --dataset lasot \
  --sequence <sequence-name>
```

Visualize predictions:

```bash
python run_all_models.py \
  --config config.local.yaml \
  --models DAM4SAM \
  --dataset didi \
  --visualize
```

## Direct Scripts

Legacy entrypoints remain available for backend-specific debugging:

```bash
python DAM4SAM/run_lasot.py --dataset_name lasot --tracker_name sam3
python DAM4SAM/run_didi.py --dataset_path /path/to/DiDi --tracker_name sam3
```

Patched direct scripts load config from `SAM3_CONFIG`, root `config.local.yaml`, or root `config.yaml`, in that order. The folder-local `config.yaml` is a compatibility template only.

## Important Files

- `tracker.py` contains the backend tracker class.
- `tracker_session.py` contains the session-mode tracker variant when available.
- `run_*.py` files are dataset-specific legacy runners.
- `vot20.py` and `vot24.py` are VOT-style entrypoints.
- `didi-workspace/` contains a small VOT toolkit workspace template.

## Notes

This backend uses the shared root `utils/` package. CUDA is expected for normal inference.
