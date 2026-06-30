<div align="center">

# SAM3-TrackBench

**A unified benchmark of SAM2-era visual trackers adapted to SAM3**

[![arXiv](https://img.shields.io/badge/arXiv-2512.22624-b31b1b.svg)](https://arxiv.org/abs/2512.22624)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

</div>

SAM3 Tracking Zoo is a unified codebase for studying visual object tracking with the SAM3 backbone. It re-implements representative SAM2-era tracking and memory designs on top of SAM3, provides standardized dataset runners, and supports controlled evaluation across multiple tracking benchmarks.

This repository accompanies the paper:

**Rethinking Memory Design in SAM-Based Visual Object Tracking**  
Mohamad Alansari, Muzammal Naseer, Hasan Al Marzouqi, Sajid Javed, Naoufel Werghi  
[arXiv:2512.22624](https://arxiv.org/abs/2512.22624)

> **Branch note:** the legacy wrapper is kept in the `v1` branch. This README documents the current `main` branch, where each SAM3 tracker adaptation is maintained as a standardized backend with matching dataset runners.

---

## Highlights

- SAM3 implementations of several SAM2-era tracking methods.
- Consistent per-backend inference scripts for common tracking benchmarks.
- Unified configuration style across tracker folders.
- Paper-oriented benchmark organization for fair comparison.
- Reproducible results layout and VOT-style workspace templates.

---

## Repository Layout

```text
SAM3_Tracking_Zoo/
|-- baseline/        # Baseline SAM3 tracker
|-- DAM4SAM-3/       # DAM4SAM memory design adapted to SAM3
|-- HiM2SAM-3/       # HiM2SAM memory design adapted to SAM3
|-- SAMITE-3/        # SAMITE memory design adapted to SAM3
|-- SAMURAI-3/       # SAMURAI memory design adapted to SAM3
|-- SAM3Long/        # SAM2Long-style long-video tracker adapted to SAM3, if included in your release
|-- assets/          # Figures and tables used by this README
`-- README.md
```

Each tracker folder follows the same high-level structure:

```text
<tracker>/
|-- sam3/                 # SAM3 model code used by the tracker
|-- utils/                # Tracker-local dataset, mask, box, and visualization helpers
|-- didi-workspace/       # VOT workspace template for DiDi-style evaluation
|-- config.yaml           # Dataset path and runtime configuration template
|-- run_didi.py           # DiDi inference
|-- run_got10k.py         # GOT-10k inference
|-- run_lasot.py          # LaSOT inference
|-- run_lasotext.py       # LaSOT Extension inference; some releases may use run_lasot_ext.py
|-- run_latot.py          # LaTOT inference
|-- run_tnl2k.py          # TNL2K inference
|-- run_trackingnet.py    # TrackingNet inference
|-- tracker.py            # Tracker implementation
|-- vot20.py              # VOT2020-style runner
`-- vot24.py              # VOTS2024-style runner
```

---

## Included Trackers

| Folder | Tracker | Source method | Description |
| --- | --- | --- | --- |
| `baseline/` | SAM3 baseline | [SAM3](https://arxiv.org/abs/2511.16719) | Baseline SAM3 tracker used as the reference backbone tracker. |
| `DAM4SAM-3/` | DAM4SAM-3 | [DAM4SAM](https://ieeexplore.ieee.org/document/11094917) | DAM4SAM-style memory design re-implemented with SAM3. |
| `SAMURAI-3/` | SAMURAI-3 | [SAMURAI](https://arxiv.org/abs/2411.11922) | SAMURAI-style motion and memory strategy adapted to SAM3. |
| `HiM2SAM-3/` | HiM2SAM-3 | [HiM2SAM](https://arxiv.org/abs/2507.07603) | Hierarchical memory tracking strategy adapted to SAM3. |
| `SAMITE-3/` | SAMITE-3 | [SAMITE](https://arxiv.org/abs/2507.21732) | SAMITE-style temporal memory mechanism adapted to SAM3. |
| `SAM3Long/` | SAM3Long | [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html) | SAM2Long-style long-video segmentation and tracking adapted to SAM3, when included in the release. |

---

## Supported Benchmarks

The paper evaluates the SAM3 adaptations across ten benchmarks:

| Benchmark | Typical runner |
| --- | --- |
| [LaSOT](https://github.com/HengLan/LaSOT_Evaluation_Toolkit) | `run_lasot.py` |
| [LaSOT Extension](https://github.com/HengLan/LaSOT_Evaluation_Toolkit) | `run_lasotext.py` or `run_lasot_ext.py` |
| [TNL2K](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit) | `run_tnl2k.py` |
| [GOT-10k](http://got-10k.aitestunion.com/) | `run_got10k.py` |
| [TrackingNet](https://huggingface.co/datasets/SilvioGiancola/TrackingNet/tree/main) | `run_trackingnet.py` |
| [DiDi](https://github.com/jovanavidenovic/DAM4SAM) | `run_didi.py` |
| [D-PTUAC](https://github.com/HamadYA/D-PTUAC) | Box/VOT-style runner, depending on the tracker release |
| [VOT2020](https://www.votchallenge.net/vot2020/) | `vot20.py` |
| [VOT2022](https://www.votchallenge.net/vot2022/) | VOT-style runner |
| [VOTS2024](https://www.votchallenge.net/vots2024/) | `vot24.py` |

Runner availability can vary slightly by tracker folder. Use the script names present in the selected tracker directory.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/HamadYA/SAM3_Tracking_Zoo.git
cd SAM3_Tracking_Zoo
```

To access the legacy wrapper described by the old README:

```bash
git checkout v1
```

For the current unified SAM3 tracking code, use:

```bash
git checkout main
```

### 2. Create an environment

The trackers are CUDA-oriented. Use a CUDA-enabled PyTorch installation that matches your system.

Example:

```bash
conda create -n sam3-zoo python=3.10 -y
conda activate sam3-zoo

# Install PyTorch for your CUDA version. Adjust the command as needed.
pip install torch torchvision

# Common runtime dependencies used by the runners.
pip install numpy scipy opencv-python pillow pyyaml tqdm pycocotools matplotlib
```

For VOT-style evaluation, install the VOT toolkit required by your benchmark setup:

```bash
pip install vot-toolkit
```

If your local release contains `environment.yml`, `requirements.txt`, or `install_env.sh`, you may use those files instead of the manual environment commands above.

### 3. Prepare SAM3 access

The tracker code expects SAM3 model code and weights to be available in the tracker folder or through the configured SAM3 builder. Make sure you have access to the required SAM3 checkpoint or Hugging Face model before running large benchmark jobs.

---

## Configuration

Each tracker directory contains a `config.yaml` file. Before running a tracker, edit the `config.yaml` inside that tracker folder and set the dataset paths for your machine.

Example:

```yaml
seed: 0

didi_dataset_path: /path/to/DiDi
lasot_dataset_path: /path/to/LaSOT
lasot_ext_dataset_path: /path/to/LaSOT_Extension
got_10k_dataset_path: /path/to/GOT-10k
tnl2k_dataset_path: /path/to/TNL2K
trackingnet_dataset_path: /path/to/TrackingNet
latot_dataset_path: /path/to/LaTOT
box_datasets_gt_masks_path: /path/to/box_dataset_gt_masks
```

Recommended path hygiene:

- Keep public config files as templates whenever possible.
- Do not commit machine-specific dataset paths, checkpoint paths, caches, or output folders.
- For paper-style comparisons, use the same dataset roots and seed across all tracker folders.
- If your checkout includes a `config.local.yaml` or `SAM3_CONFIG` loader, prefer that local config workflow for private paths.

---

## Running Inference

Run commands from inside the tracker folder you want to evaluate.

### LaSOT example

```bash
cd baseline
python run_lasot.py
```

### TNL2K example

```bash
cd DAM4SAM-3
python run_tnl2k.py
```

### DiDi example

```bash
cd SAMURAI-3
python run_didi.py
```

### VOT/VOTS-style examples

```bash
cd HiM2SAM-3
python vot20.py
```

```bash
cd SAMITE-3
python vot24.py
```

### Run the same benchmark across multiple trackers

From the repository root:

```bash
for tracker in baseline DAM4SAM-3 SAMURAI-3 HiM2SAM-3 SAMITE-3; do
  echo "Running ${tracker} on LaSOT"
  (cd "${tracker}" && python run_lasot.py)
done
```

For a large benchmark run, verify one sequence or one short dataset first, then launch the full evaluation.

---

## Outputs

Most runners write predictions under an output directory such as:

```text
<tracker>/out/<tracker-name>/<dataset>/<run-id>/
```

The exact output layout can differ slightly across benchmark scripts. Use the official evaluation toolkit for each dataset to compute final metrics from the generated prediction files.

Generated outputs, caches, and checkpoints should not be committed to the repository.

---

## Results

The paper-level result table can be shown from the repository assets:

```markdown
![Performance table](assets/table1.png)
```

![Performance table](assets/table1.png)

Additional logs and benchmark artifacts can be released separately to keep the repository lightweight.

---

## Reproducibility Checklist

Before reporting numbers, verify that:

- The same SAM3 checkpoint or model source is used for every tracker.
- Dataset roots are correct in each tracker folder.
- The same random seed is used across trackers.
- Existing prediction files are removed or intentionally reused.
- Each benchmark is evaluated with its official toolkit.
- VOT/VOTS workspaces contain only templates unless the benchmark data has been locally installed.

---

## Troubleshooting

### Dataset path is empty or not found

Open the selected tracker folder and edit its `config.yaml`. The scripts read dataset paths from that file.

### A sequence is skipped

Some runners skip sequences when the expected output file already exists. Remove the old prediction file or use a new output directory before re-running.

### CUDA or out-of-memory error

These trackers are intended for CUDA execution. Use a GPU with sufficient memory, close other GPU jobs, or run one benchmark/tracker at a time.

### VOT toolkit import error

Install the VOT toolkit in the active environment:

```bash
pip install vot-toolkit
```

### DiDi/VOT workspace confusion

The `didi-workspace/` directories are templates for VOT-style workflows. They do not include datasets or generated result files.

---

## Relationship to the Paper

The paper studies how memory mechanisms from SAM2-based trackers transfer to SAM3. This repository provides the SAM3 tracking zoo used for those comparisons and is intended to make the adaptations inspectable and reproducible.

The arXiv preprint notes that some results are being finalized and may be updated in a future revision. Please check the paper and repository for the latest released numbers before citing specific results.

---

## Acknowledgments

This project builds on high-quality open-source work from the SAM and visual object tracking communities. We thank the authors of SAM3, SAM2, DAM4SAM, SAMURAI, HiM2SAM, SAMITE, and SAM2Long for making their methods, models, and code available to the community.

The tracker adaptations in this repository follow the original method designs as closely as possible while using a standardized SAM3-based inference and evaluation pipeline.

---

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{alansari2025rethinking,
  title={Rethinking Memory Design in SAM-Based Visual Object Tracking},
  author={Alansari, Mohamad and Naseer, Muzammal and Al Marzouqi, Hasan and Werghi, Naoufel and Javed, Sajid},
  journal={arXiv preprint arXiv:2512.22624},
  year={2025}
}
```

Please also cite the original methods used in your experiments, including SAM3, DAM4SAM, SAMURAI, HiM2SAM, SAMITE, SAM2Long, and the relevant dataset papers or benchmark toolkits.

---

## License

This project is intended to be released under the Apache 2.0 License. Please make sure the repository includes the corresponding `LICENSE` file before public release or redistribution.

Third-party tracker code, model code, checkpoints, and datasets may have their own licenses. Users are responsible for following the license terms of all upstream projects and datasets.
