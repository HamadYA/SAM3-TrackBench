import argparse
import os
import random
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sam3_tool_utils import (  # noqa: E402
    add_sam3_mask_prompt,
    autocast_context,
    build_sam3_tracker_predictor,
    iter_sam3_propagation,
    mask_array_for_object,
    set_sam3_memory_selection_params,
)
from vot_dataset_utils import DATASET_REGISTRY, SequenceRecord, build_dataset  # noqa: E402

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "vot_config.yaml")


def split_list(lst, n):
    """Split a list into n roughly equal non-empty chunks."""
    if n <= 0:
        raise ValueError("number of chunks must be positive")
    avg = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    for i in range(n):
        chunk_size = avg + (1 if i < remainder else 0)
        chunks.append(lst[start : start + chunk_size])
        start += chunk_size
    return chunks


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        raise ValueError(f"chunk index {k} is out of range for {len(chunks)} chunks")
    return chunks[k]


def bbox_xywh_to_xyxy(bbox, width, height):
    x, y, w, h = [float(v) for v in bbox[:4]]
    x0 = max(0.0, min(x, width - 1))
    y0 = max(0.0, min(y, height - 1))
    x1 = max(x0 + 1.0, min(x + max(w, 1.0), width))
    y1 = max(y0 + 1.0, min(y + max(h, 1.0), height))
    return [x0, y0, x1, y1]


def mask_to_bbox(mask):
    mask = np.asarray(mask) > 0
    if mask.sum() == 0:
        return [0, 0, 0, 0]
    x_idxs = np.where(mask.sum(axis=0) > 0)[0]
    y_idxs = np.where(mask.sum(axis=1) > 0)[0]
    x0 = int(x_idxs.min())
    x1 = int(x_idxs.max())
    y0 = int(y_idxs.min())
    y1 = int(y_idxs.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def save_boxes(file_path, bboxes):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for bbox in bboxes:
            f.write("{:.2f},{:.2f},{:.2f},{:.2f}\n".format(*bbox))


def save_mask_png(file_path, mask):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    output = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    output.save(file_path)


def decode_vot_mask_line(line, output_width, output_height):
    line = line.strip()
    if not line.startswith("m"):
        raise ValueError("Only VOT mask regions starting with 'm' are supported")
    values = [int(v) for v in line[1:].split(",") if v]
    if len(values) < 5:
        raise ValueError(f"Invalid VOT mask region: {line[:80]}")
    x, y, width, height = values[:4]
    counts = values[4:]
    flat = np.zeros(width * height, dtype=np.uint8)
    index = 0
    value = 0
    for count in counts:
        next_index = index + count
        if value:
            flat[index:next_index] = 1
        index = next_index
        value = 1 - value
    crop = flat.reshape((height, width))

    mask = np.zeros((output_height, output_width), dtype=bool)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(output_width, x + width)
    y1 = min(output_height, y + height)
    if x1 > x0 and y1 > y0:
        crop_x0 = x0 - x
        crop_y0 = y0 - y
        mask[y0:y1, x0:x1] = crop[
            crop_y0 : crop_y0 + (y1 - y0), crop_x0 : crop_x0 + (x1 - x0)
        ]
    return mask


def encode_vot_mask(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = mask.shape
    flat = mask.ravel()
    counts = []
    current_value = 0
    count = 0
    for value in flat:
        value = int(value)
        if value == current_value:
            count += 1
        else:
            counts.append(count)
            current_value = value
            count = 1
    counts.append(count)
    counts_text = ",".join(str(c) for c in counts)
    return f"m0,0,{width},{height},{counts_text}"


def load_vot_mask(mask_path, width, height):
    with open(mask_path, "r") as f:
        first_line = next((line for line in f if line.strip()), "")
    if first_line.lstrip().startswith("m"):
        return decode_vot_mask_line(first_line, width, height)

    try:
        from vot.region.io import read_trajectory
    except ImportError as exc:
        raise RuntimeError("The DIDI dataset requires a VOT mask text file") from exc
    mask = read_trajectory(mask_path)[0].rasterize((0, 0, width - 1, height - 1))
    return np.asarray(mask).astype(bool)


def save_vot_mask_trajectory(sequence, masks, output_dir):
    sequence_dir = os.path.join(output_dir, sequence.name)
    os.makedirs(sequence_dir, exist_ok=True)
    trajectory_path = os.path.join(sequence_dir, f"{sequence.name}.txt")
    with open(trajectory_path, "w") as f:
        for mask in masks:
            f.write(encode_vot_mask(mask) + "\n")
    return trajectory_path


def load_config(config_file):
    if config_file is None:
        return {}
    if not os.path.exists(config_file):
        if os.path.abspath(config_file) == os.path.abspath(DEFAULT_CONFIG_FILE):
            return {}
        raise FileNotFoundError(f"Config file not found: {config_file}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read config YAML files") from exc
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def resolve_model_config(args, config):
    model_name = args.model_name or config.get("default_model")
    model_config = {}
    if model_name:
        model_config = (config.get("models") or {}).get(model_name, {})
        if not model_config:
            available = ", ".join(sorted((config.get("models") or {}).keys()))
            raise ValueError(
                f"Model {model_name!r} not found in config. Available: {available}"
            )

    checkpoint_path = (
        args.sam3_checkpoint
        or model_config.get("sam3_checkpoint")
        or model_config.get("checkpoint")
    )
    load_from_hf = bool(model_config.get("load_from_hf", checkpoint_path is None))
    if args.disable_hf_download:
        load_from_hf = False
    if not load_from_hf and checkpoint_path is None:
        raise ValueError(
            "A SAM3 checkpoint is required when Hugging Face download is disabled"
        )
    compile_model = bool(model_config.get("compile", False)) or args.compile_model
    apply_temporal_disambiguation = bool(
        model_config.get("apply_temporal_disambiguation", True)
    )
    if args.disable_temporal_disambiguation:
        apply_temporal_disambiguation = False
    return (
        checkpoint_path,
        load_from_hf,
        compile_model,
        apply_temporal_disambiguation,
    )


def resolve_dataset_dir(args, config):
    if args.dataset_dir:
        return args.dataset_dir
    datasets = config.get("datasets") or {}
    dataset_config = datasets.get(args.dataset_name)
    if isinstance(dataset_config, dict):
        return dataset_config.get("path") or dataset_config.get("dataset_dir")
    return dataset_config


def resolve_output_dir(args, config):
    if args.output_dir:
        return args.output_dir
    outputs = config.get("outputs") or {}
    output_config = outputs.get(args.dataset_name)
    if isinstance(output_config, dict):
        return output_config.get("path") or output_config.get("output_dir")
    return output_config


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_numeric_frame_dir(frame_paths, temp_root):
    """Create numeric symlinks because SAM3 sorts frame basenames as integers."""
    frame_dir = os.path.join(temp_root, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for idx, src in enumerate(frame_paths):
        ext = os.path.splitext(src)[1]
        if ext.lower() not in [".jpg", ".jpeg"]:
            ext = ".jpg"
        dst = os.path.join(frame_dir, f"{idx:08d}{ext.lower()}")
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            shutil.copy2(src, dst)
    return frame_dir


class TrackingVisualizer:
    def __init__(self, window_name="SAM3 VOT", wait_ms=30, max_size=1280):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --visualize") from exc
        self.cv2 = cv2
        self.window_name = window_name
        self.wait_ms = wait_ms
        self.max_size = max_size
        self.paused = wait_ms == 0
        self.window_created = False

    def _overlay_mask(self, image, mask):
        mask = np.asarray(mask) > 0
        if not mask.any():
            return image
        color = np.array([255, 255, 0], dtype=np.float32)
        image_float = image.astype(np.float32)
        image_float[mask] = 0.45 * image_float[mask] + 0.55 * color
        image = image_float.astype(np.uint8)
        contours, _ = self.cv2.findContours(
            mask.astype(np.uint8), self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        self.cv2.drawContours(image, contours, -1, (255, 255, 0), 2)
        return image

    def _overlay_box(self, image, bbox):
        x, y, w, h = [int(round(v)) for v in bbox]
        if w > 0 and h > 0:
            self.cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        return image

    def render(self, frame_path, mask, bbox, sequence_name, frame_idx):
        image = np.asarray(Image.open(frame_path).convert("RGB")).copy()
        image = self._overlay_mask(image, mask)
        image = self._overlay_box(image, bbox)
        title = f"{sequence_name}  frame {frame_idx}"
        self.cv2.putText(
            image,
            title,
            (12, 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        height, width = image.shape[:2]
        scale = min(1.0, self.max_size / max(height, width))
        if scale < 1.0:
            image = self.cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR)

    def show(self, frame_path, mask, bbox, sequence_name, frame_idx):
        if not self.window_created:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_AUTOSIZE)
            self.window_created = True
        image = self.render(frame_path, mask, bbox, sequence_name, frame_idx)
        self.cv2.imshow(self.window_name, image)
        wait = 0 if self.paused else self.wait_ms
        key = self.cv2.waitKey(wait)
        if key == 27:
            raise KeyboardInterrupt
        if key == 32:
            self.paused = not self.paused


@torch.inference_mode()
def run_sequence(
    predictor,
    sequence: SequenceRecord,
    score_thresh=0.0,
    num_pathway=3,
    iou_thre=0.1,
    uncertainty=2,
):
    with tempfile.TemporaryDirectory(prefix="sam3_vot_") as temp_root:
        video_dir = make_numeric_frame_dir(sequence.frame_paths, temp_root)
        inference_state = predictor.init_state(
            video_path=video_dir, async_loading_frames=False
        )
        set_sam3_memory_selection_params(
            inference_state,
            num_pathway=num_pathway,
            iou_thre=iou_thre,
            uncertainty=uncertainty,
        )

        height = inference_state["video_height"]
        width = inference_state["video_width"]

        init_mask = None
        if sequence.init_mask_path is not None:
            init_mask = load_vot_mask(sequence.init_mask_path, width, height)
            add_sam3_mask_prompt(predictor, inference_state, 0, 1, init_mask)
        else:
            init_box = np.asarray(
                bbox_xywh_to_xyxy(sequence.init_bbox, width, height),
                dtype=np.float32,
            )
            init_box[0] /= width
            init_box[1] /= height
            init_box[2] /= width
            init_box[3] /= height
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=init_box[None],
            )

        predictions = []
        masks = []
        for frame_idx, obj_ids, frame_masks in iter_sam3_propagation(
            predictor,
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=None,
            reverse=False,
            propagate_preflight=True,
        ):
            mask = mask_array_for_object(obj_ids, frame_masks, 1, score_thresh)
            if mask is None:
                mask = np.zeros((height, width), dtype=bool)
            if frame_idx == 0 and sequence.init_bbox is not None:
                bbox = [float(v) for v in sequence.init_bbox[:4]]
            else:
                if frame_idx == 0 and init_mask is not None:
                    mask = init_mask
                bbox = mask_to_bbox(mask)
            predictions.append(bbox)
            masks.append(mask)

    return predictions, masks


def save_sequence_outputs(sequence, predictions, masks, output_dir, output_mask_dir=None):
    output_path = os.path.join(output_dir, f"{sequence.name}.txt")
    save_boxes(output_path, predictions)
    if sequence.init_mask_path is not None:
        save_vot_mask_trajectory(sequence, masks, output_dir)
    if output_mask_dir is not None:
        for frame_idx, mask in enumerate(masks):
            mask_path = os.path.join(output_mask_dir, sequence.name, f"{frame_idx:08d}.png")
            save_mask_png(mask_path, mask)
    return output_path


def visualize_sequence(sequence, predictions, masks, visualizer, vis_output_dir=None):
    if vis_output_dir is not None:
        os.makedirs(os.path.join(vis_output_dir, sequence.name), exist_ok=True)
    for frame_idx, (frame_path, bbox, mask) in enumerate(
        zip(sequence.frame_paths, predictions, masks)
    ):
        if vis_output_dir is not None:
            rendered = visualizer.render(frame_path, mask, bbox, sequence.name, frame_idx)
            out_path = os.path.join(vis_output_dir, sequence.name, f"{frame_idx:08d}.jpg")
            visualizer.cv2.imwrite(out_path, rendered)
        visualizer.show(frame_path, mask, bbox, sequence.name, frame_idx)


def build_parser(default_dataset_name=None):
    parser = argparse.ArgumentParser(
        description="Run SAM3 single-object tracking inference on VOT-style datasets."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="YAML config file with model checkpoints and dataset paths",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="model entry to use from --config_file",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=None,
        help="local SAM3 checkpoint path; if omitted, the builder may download the default checkpoint",
    )
    parser.add_argument(
        "--disable_hf_download",
        action="store_true",
        help="do not download the default SAM3 checkpoint when --sam3_checkpoint is omitted",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="enable torch.compile when building the SAM3 model",
    )
    parser.add_argument(
        "--disable_temporal_disambiguation",
        default=False,
        action="store_true",
        help="disable SAM3 temporal disambiguation / memory selection",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=default_dataset_name,
        choices=sorted(DATASET_REGISTRY),
        required=default_dataset_name is None,
        help="tracking dataset to run",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="root directory of the selected tracking dataset; overrides config file",
    )
    parser.add_argument(
        "--sequence_list_file",
        type=str,
        default=None,
        help="optional text file containing sequence names to run",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="optional single sequence name to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save one prediction txt file per sequence",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        default=None,
        help="optional directory to save propagated binary masks as PNG files",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for converting output mask logits to binary masks",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="deprecated SAM2 option retained for old commands; ignored by SAM3",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing sequence txt outputs",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="visualize predictions instead of requiring output txt files",
    )
    parser.add_argument(
        "--visualize_wait_ms",
        type=int,
        default=30,
        help="OpenCV wait time per frame; use 0 to pause on every frame",
    )
    parser.add_argument(
        "--visualize_max_size",
        type=int,
        default=1280,
        help="longest display side for visualization",
    )
    parser.add_argument(
        "--visualize_output_dir",
        type=str,
        default=None,
        help="optional directory to save rendered visualization frames",
    )
    parser.add_argument(
        "--num_pathway",
        type=int,
        default=3,
        help="number of SAM3 memory-selection pathways to maintain",
    )
    parser.add_argument(
        "--iou_thre",
        type=float,
        default=0.1,
        help="IoU threshold used by SAM3 memory selection",
    )
    parser.add_argument(
        "--uncertainty",
        type=float,
        default=2,
        help="uncertainty threshold used by SAM3 pathway selection",
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="index of this worker's chunk",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="number of chunks per node",
    )
    parser.add_argument(
        "--node_id",
        type=int,
        default=0,
        help="index of the current node",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="total number of nodes",
    )
    return parser


def main(default_dataset_name=None):
    parser = build_parser(default_dataset_name=default_dataset_name)
    args = parser.parse_args()
    config = load_config(args.config_file)
    set_seed(config.get("seed"))

    try:
        (
            args.sam3_checkpoint,
            args.load_from_hf,
            args.compile_model,
            args.apply_temporal_disambiguation,
        ) = resolve_model_config(args, config)
    except ValueError as exc:
        parser.error(str(exc))

    args.dataset_dir = resolve_dataset_dir(args, config)
    if not args.dataset_dir:
        parser.error(
            "--dataset_dir is required unless the selected dataset has a path in --config_file"
        )

    cli_output_dir = args.output_dir
    if args.visualize and cli_output_dir is None:
        args.output_dir = None
    else:
        args.output_dir = resolve_output_dir(args, config)
    if not args.visualize and not args.output_dir:
        parser.error("--output_dir is required unless --visualize is set")

    dataset = build_dataset(
        args.dataset_name,
        args.dataset_dir,
        sequence_list_file=args.sequence_list_file,
    )
    sequence_names = [args.sequence] if args.sequence else list(dataset.sequence_list)
    global_chunk_id = args.node_id * args.num_chunks + args.chunk_id
    sequence_names = get_chunk(
        sequence_names, args.num_nodes * args.num_chunks, global_chunk_id
    )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.output_mask_dir is not None:
        os.makedirs(args.output_mask_dir, exist_ok=True)
    if args.visualize_output_dir is not None:
        os.makedirs(args.visualize_output_dir, exist_ok=True)
    visualizer = None
    if args.visualize:
        print(
            "visualization window will open after model loading and sequence "
            "propagation finish"
        )
        visualizer = TrackingVisualizer(
            wait_ms=args.visualize_wait_ms,
            max_size=args.visualize_max_size,
        )

    predictor = build_sam3_tracker_predictor(
        checkpoint_path=args.sam3_checkpoint,
        load_from_hf=args.load_from_hf,
        compile_model=args.compile_model,
        apply_temporal_disambiguation=args.apply_temporal_disambiguation,
    )

    print(
        f"running SAM3 VOT inference on {len(sequence_names)} "
        f"{args.dataset_name} sequences:\n{sequence_names}"
    )
    for n_seq, sequence_name in enumerate(sequence_names):
        output_path = (
            os.path.join(args.output_dir, f"{sequence_name}.txt")
            if args.output_dir is not None
            else None
        )
        if output_path is not None and os.path.exists(output_path) and not args.overwrite:
            print(f"{sequence_name} already exists, skipping")
            continue

        print(f"\n{n_seq + 1}/{len(sequence_names)} - running on {sequence_name}")
        try:
            sequence = dataset.get_sequence(sequence_name)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"skipping {sequence_name}: {exc}")
            continue

        print(
            f"propagating {len(sequence.frame_paths)} frames; "
            "visualization starts after this sequence is ready"
        )
        with autocast_context():
            predictions, masks = run_sequence(
                predictor=predictor,
                sequence=sequence,
                score_thresh=args.score_thresh,
                num_pathway=args.num_pathway,
                iou_thre=args.iou_thre,
                uncertainty=args.uncertainty,
            )
        if visualizer is not None:
            print(f"visualizing {sequence.name}")
            visualize_sequence(
                sequence,
                predictions,
                masks,
                visualizer,
                vis_output_dir=args.visualize_output_dir,
            )
        if args.output_dir is not None:
            saved_path = save_sequence_outputs(
                sequence,
                predictions,
                masks,
                output_dir=args.output_dir,
                output_mask_dir=args.output_mask_dir,
            )
            print(f"results saved to: {saved_path}")

    if args.output_dir is None:
        print(f"completed SAM3 VOT visualization on {len(sequence_names)} sequences")
    else:
        print(
            f"completed SAM3 VOT inference on {len(sequence_names)} sequences -- "
            f"outputs saved to {args.output_dir}"
        )


if __name__ == "__main__":
    main()
