# SAM3 Tracking Zoo
A collection of SAM2-era tracking methods re-implemented on top of SAM3.
This repository provides a unified structure and inference interface for multiple trackers adapted to the SAM3 backbone.

# Repository Structure
```
baseline/        # Baseline SAM3 tracker
DAM4SAM-3/       # DAM4SAM adapted to SAM3
SAMURAI-3/       # SAMURAI adapted to SAM3
HiM2SAM-3/       # HiM2SAM adapted to SAM3
SAMITE-3/        # SAMITE adapted to SAM3
SAM3Long/        # SAM2Long adapted to SAM3
```

Each directory contains:
- Model implementation
- Inference scripts for all supported datasets

# Included Trackers (SAM2 → SAM3 Adaptations)
| Tracker       | Origin (SAM2) | Description                                       |
| ------------- | ------------- | ------------------------------------------------- |
| **baseline**  | [SAM3](https://arxiv.org/abs/2511.16719) | SAM3 tracker.            |
| **DAM4SAM-3** | [DAM4SAM](https://ieeexplore.ieee.org/document/11094917)       | DAM4SAM applied to SAM3.    |
| **SAMURAI-3** | [SAMURAI](https://arxiv.org/abs/2411.11922)       | SAMURAI applied to SAM3. |
| **HiM2SAM-3** | [HiM2SAM](https://arxiv.org/abs/2507.07603)       | HiM2SAM applied to SAM3.    |
| **SAMITE-3**  | [SAMITE](https://arxiv.org/abs/2507.21732)        | SAMITE applied to SAM3. |
| **SAM3Long**  | [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html)      | SAM2Long applied to SAM3.     |


# Inference
All trackers include standardized inference scripts covering 10 benchmarks.

Supported Datasets:
- [LaSOT](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)
- [LaSOT Extension](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)
- [TNL2K](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit)
- [GOT-10k](http://got-10k.aitestunion.com/)
- [TrackingNet](https://huggingface.co/datasets/SilvioGiancola/TrackingNet/tree/main)
- [DiDi](https://github.com/jovanavidenovic/DAM4SAM)
- [D-PTUAC](https://github.com/HamadYA/D-PTUAC)
- [VOT2020](https://www.votchallenge.net/vot2020/)
- [VOT2022](https://www.votchallenge.net/vot2022/)
- [VOTS2024](https://www.votchallenge.net/vots2024/)

You need to change the dataset path in each **config.yaml** and then run each command.
Example Command:
```
python run_lasot.py
```

