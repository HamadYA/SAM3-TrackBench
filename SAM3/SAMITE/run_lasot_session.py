from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from unified.config import load_config

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from tracker_session import SAMSessionTracker
from utils.dataset_utils import get_dataset, pil2array
from utils.mask_utils import save_boxes
from utils.visualization_utils import VisualizerTracking


config = load_config(__file__)

seed = config["seed"]
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def _sequence_frame_dir(dataset, sequence_name):
    if sequence_name not in dataset.sequences:
        dataset.get_seq_len(sequence_name)
    frames = dataset.sequences[sequence_name]["frames"]
    if not frames:
        raise RuntimeError(f"No frames found for sequence {sequence_name}")
    return os.path.dirname(frames[0])


@torch.inference_mode()
def main(
    tracker_name,
    dataset_name,
    output_dir,
    selected_sequence=None,
    output_prob_thresh=0.5,
    compile_model=False,
    warm_up=False,
    max_num_objects=16,
    multiplex_count=16,
    use_fa3=False,
    use_rope_real=True,
    async_loading_frames=True,
):
    dataset = get_dataset(dataset_name, init_masks=None)
    sequences = dataset.sequence_list

    tracker = SAMSessionTracker(
        tracker_name=tracker_name,
        output_prob_thresh=output_prob_thresh,
        compile_model=compile_model,
        warm_up=warm_up,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        async_loading_frames=async_loading_frames,
    )

    try:
        for sequence_name in sequences:
            if selected_sequence is not None and selected_sequence != sequence_name:
                continue

            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{sequence_name}.txt")
                if os.path.exists(output_path):
                    print(f"{sequence_name} has already been processed. Skipping...")
                    continue
            else:
                visualizer = VisualizerTracking()

            seq_len = dataset.get_seq_len(sequence_name)
            if seq_len == 0:
                continue

            first_img = dataset.get_pil_frame(sequence_name, 0)
            init_bbox = [float(v) for v in dataset.get_groundtruth(sequence_name, 0)]
            sequence_dir = _sequence_frame_dir(dataset, sequence_name)

            print(
                f"Processing sequence: {sequence_name} with {seq_len} frames "
                f"using {tracker.tracker_name} session mode."
            )

            predictions = [init_bbox]
            tracker.initialize(
                resource_path=sequence_dir,
                bbox=init_bbox,
                image_size=first_img.size,
            )

            if output_dir is None:
                visualizer.show(pil2array(first_img), box=init_bbox)

            try:
                for frame_idx in range(1, seq_len):
                    outputs = tracker.track(frame_idx)
                    pred_bbox = outputs["pred_bbox"]
                    if pred_bbox is None:
                        pred_bbox = [0, 0, 0, 0]
                    predictions.append(pred_bbox)

                    if output_dir is None:
                        img = dataset.get_pil_frame(sequence_name, frame_idx)
                        visualizer.show(pil2array(img), box=pred_bbox)
            finally:
                tracker.close()

            if output_dir is not None:
                output_path = os.path.join(output_dir, f"{sequence_name}.txt")
                save_boxes(output_path, predictions)
                print("Results saved to:", output_path)
    finally:
        tracker.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lasot",
        help="got_10k | lasot | lasot_ext | trackingnet | tnl2k | latot | otb",
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="sam3.1",
        help="sam3 | sam3.1",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--output_prob_thresh", type=float, default=0.5)
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--warm_up", action="store_true")
    parser.add_argument("--max_num_objects", type=int, default=16)
    parser.add_argument("--multiplex_count", type=int, default=16)
    parser.add_argument("--use_fa3", action="store_true")
    parser.add_argument("--no_rope_real", dest="use_rope_real", action="store_false")
    parser.add_argument(
        "--no_async_loading_frames",
        dest="async_loading_frames",
        action="store_false",
    )
    parser.set_defaults(use_rope_real=True, async_loading_frames=True)

    args = parser.parse_args()

    if args.output_dir is not None:
        base_output_dir = os.path.join(args.output_dir, args.tracker_name)
        run_idx = 0
        output_dir = os.path.join(base_output_dir, args.dataset_name, f"{run_idx:03d}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    main(
        tracker_name=args.tracker_name,
        dataset_name=args.dataset_name,
        output_dir=output_dir,
        selected_sequence=args.sequence,
        output_prob_thresh=args.output_prob_thresh,
        compile_model=args.compile_model,
        warm_up=args.warm_up,
        max_num_objects=args.max_num_objects,
        multiplex_count=args.multiplex_count,
        use_fa3=args.use_fa3,
        use_rope_real=args.use_rope_real,
        async_loading_frames=args.async_loading_frames,
    )
