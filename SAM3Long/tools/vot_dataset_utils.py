import glob
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG")


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    frame_paths: list[str]
    gt_path: str
    init_bbox: Optional[list[float]] = None
    init_mask_path: Optional[str] = None


def parse_bbox_line(line: str) -> list[float]:
    parts = [p for p in re.split(r"[, \t]+", line.strip()) if p]
    if len(parts) < 4:
        raise ValueError(f"Expected at least 4 bbox values, got: {line!r}")
    return [float(v) for v in parts[:4]]


def load_first_bbox(gt_path: str) -> list[float]:
    with open(gt_path, "r") as f:
        for line in f:
            if line.strip():
                return parse_bbox_line(line)
    raise ValueError(f"No bbox annotation found in {gt_path}")


def image_sort_key(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    numbers = re.findall(r"\d+", name)
    if numbers:
        return (0, [int(n) for n in numbers], name)
    return (1, [], name)


def list_images(frame_dir: str) -> list[str]:
    frame_paths = [
        p
        for p in glob.glob(os.path.join(frame_dir, "*"))
        if os.path.splitext(p)[1] in IMAGE_EXTENSIONS
    ]
    return sorted(frame_paths, key=image_sort_key)


def read_sequence_names(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def read_sequence_list(
    base_dir: str,
    sequence_list_file: Optional[str],
    default_list_file: Optional[str],
    fallback: Callable[[], list[str]],
) -> list[str]:
    if sequence_list_file is not None:
        return read_sequence_names(sequence_list_file)
    if default_list_file is not None:
        path = os.path.join(base_dir, default_list_file)
        if os.path.exists(path):
            return read_sequence_names(path)
    return fallback()


class BoxTrackingDataset:
    default_list_file: Optional[str] = "list.txt"

    def __init__(self, base_dir: str, sequence_list_file: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir)
        self.sequence_list = read_sequence_list(
            self.base_dir,
            sequence_list_file,
            self.default_list_file,
            self._fallback_sequence_list,
        )

    def _fallback_sequence_list(self) -> list[str]:
        return sorted(
            p
            for p in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, p))
        )

    def frame_dir(self, sequence_name: str) -> str:
        raise NotImplementedError

    def gt_path(self, sequence_name: str) -> str:
        raise NotImplementedError

    def get_sequence(self, sequence_name: str) -> SequenceRecord:
        frame_dir = self.frame_dir(sequence_name)
        gt_path = self.gt_path(sequence_name)
        if not os.path.isdir(frame_dir):
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

        frame_paths = list_images(frame_dir)
        if not frame_paths:
            raise RuntimeError(f"No JPEG frames found in {frame_dir}")

        return SequenceRecord(
            name=sequence_name,
            frame_paths=frame_paths,
            gt_path=gt_path,
            init_bbox=load_first_bbox(gt_path),
        )


class GOT10KDataset(BoxTrackingDataset):
    def frame_dir(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name)

    def gt_path(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, "groundtruth.txt")


class LaSOTDataset(BoxTrackingDataset):
    default_list_file = "testing_set.txt"

    def _fallback_sequence_list(self) -> list[str]:
        sequences = []
        for category in sorted(os.listdir(self.base_dir)):
            category_dir = os.path.join(self.base_dir, category)
            if not os.path.isdir(category_dir):
                continue
            for seq in sorted(os.listdir(category_dir)):
                if os.path.isdir(os.path.join(category_dir, seq, "img")):
                    sequences.append(seq)
        return sequences

    def _category(self, sequence_name: str) -> str:
        return sequence_name.split("-")[0]

    def frame_dir(self, sequence_name: str) -> str:
        return os.path.join(
            self.base_dir, self._category(sequence_name), sequence_name, "img"
        )

    def gt_path(self, sequence_name: str) -> str:
        return os.path.join(
            self.base_dir, self._category(sequence_name), sequence_name, "groundtruth.txt"
        )


class LaSOTExtDataset(LaSOTDataset):
    pass


class TrackingNetDataset(BoxTrackingDataset):
    def _fallback_sequence_list(self) -> list[str]:
        frames_root = os.path.join(self.base_dir, "frames")
        return sorted(
            p
            for p in os.listdir(frames_root)
            if os.path.isdir(os.path.join(frames_root, p))
        )

    def frame_dir(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, "frames", sequence_name)

    def gt_path(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, "anno", f"{sequence_name}.txt")


class TNL2KDataset(BoxTrackingDataset):
    def frame_dir(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, "imgs")

    def gt_path(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, "groundtruth.txt")


class LaToTDataset(BoxTrackingDataset):
    def frame_dir(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, "img")

    def gt_path(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, f"{sequence_name}.txt")


class OTBDataset(BoxTrackingDataset):
    def frame_dir(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, "img")

    def gt_path(self, sequence_name: str) -> str:
        return os.path.join(self.base_dir, sequence_name, "groundtruth_rect.txt")


class DidiDataset(BoxTrackingDataset):
    def __init__(self, base_dir: str, sequence_list_file: Optional[str] = None):
        self._vot_dataset = None
        super().__init__(base_dir, sequence_list_file=sequence_list_file)

    def _fallback_sequence_list(self) -> list[str]:
        sequences_root = os.path.join(self.base_dir, "sequences")
        if os.path.isdir(sequences_root):
            return sorted(
                p
                for p in os.listdir(sequences_root)
                if os.path.isdir(os.path.join(sequences_root, p))
            )
        return super()._fallback_sequence_list()

    def _load_vot_dataset(self):
        if self._vot_dataset is None:
            try:
                from vot.dataset import load_dataset
            except ImportError as exc:
                raise RuntimeError(
                    "The DIDI dataset loader requires the VOT toolkit package"
                ) from exc
            self._vot_dataset = load_dataset(self.base_dir)
        return self._vot_dataset

    def _sequence_root(self, sequence_name: str) -> str:
        direct_root = os.path.join(self.base_dir, sequence_name)
        if os.path.isdir(direct_root):
            return direct_root
        workspace_root = os.path.join(self.base_dir, "sequences", sequence_name)
        if os.path.isdir(workspace_root):
            return workspace_root
        raise FileNotFoundError(f"DIDI sequence directory not found: {sequence_name}")

    def _get_sequence_without_vot(self, sequence_name: str) -> SequenceRecord:
        sequence_root = self._sequence_root(sequence_name)
        frame_dir = os.path.join(sequence_root, "color")
        if not os.path.isdir(frame_dir):
            frame_dir = os.path.join(sequence_root, "img")
        if not os.path.isdir(frame_dir):
            frame_dir = sequence_root
        frame_paths = list_images(frame_dir)
        if not frame_paths:
            raise RuntimeError(f"No JPEG frames found in DIDI sequence {sequence_name}")

        init_mask_path = os.path.join(sequence_root, "first_frame_segm.txt")
        if not os.path.exists(init_mask_path):
            raise FileNotFoundError(f"Initial DIDI mask not found: {init_mask_path}")

        return SequenceRecord(
            name=sequence_name,
            frame_paths=frame_paths,
            gt_path=init_mask_path,
            init_mask_path=init_mask_path,
        )

    def get_sequence(self, sequence_name: str) -> SequenceRecord:
        try:
            dataset = self._load_vot_dataset()
        except RuntimeError:
            return self._get_sequence_without_vot(sequence_name)
        sequence = dataset[sequence_name]
        frame_paths = [sequence.frame(i).filename() for i in range(len(sequence))]
        if not frame_paths:
            raise RuntimeError(f"No frames found in DIDI sequence {sequence_name}")

        sequence_root = sequence.metadata("root")
        init_mask_path = os.path.join(sequence_root, "first_frame_segm.txt")
        if not os.path.exists(init_mask_path):
            raise FileNotFoundError(f"Initial DIDI mask not found: {init_mask_path}")

        return SequenceRecord(
            name=sequence_name,
            frame_paths=frame_paths,
            gt_path=init_mask_path,
            init_mask_path=init_mask_path,
        )


DATASET_REGISTRY = {
    "didi": DidiDataset,
    "got_10k": GOT10KDataset,
    "lasot": LaSOTDataset,
    "lasot_ext": LaSOTExtDataset,
    "trackingnet": TrackingNetDataset,
    "tnl2k": TNL2KDataset,
    "latot": LaToTDataset,
    "otb": OTBDataset,
}


def build_dataset(
    dataset_name: str, dataset_dir: str, sequence_list_file: Optional[str] = None
) -> BoxTrackingDataset:
    try:
        dataset_cls = DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset {dataset_name!r}. Valid options: {valid}") from exc
    return dataset_cls(dataset_dir, sequence_list_file=sequence_list_file)
