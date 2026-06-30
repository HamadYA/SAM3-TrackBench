from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from unified.config import load_config

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from sam3.model_builder import build_sam3_predictor
from utils.mask_utils import mask2box


config = load_config(__file__)

seed = config["seed"]
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


class SAMSessionTracker:
    """Session-mode SAM3/SAM3.1 tracker with frame-by-frame access."""

    def __init__(
        self,
        tracker_name="sam3.1",
        output_prob_thresh=0.5,
        obj_id=1,
        compile_model=False,
        warm_up=False,
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        async_loading_frames=True,
        checkpoint_path=None,
    ):
        self.tracker_name = self._normalize_tracker_name(tracker_name)
        self.output_prob_thresh = output_prob_thresh
        self.requested_obj_id = obj_id
        self.session_id = None
        self.target_obj_id = None
        self.frame_outputs = {}
        self.img_width = None
        self.img_height = None
        self.init_bbox = None
        self._forward_propagated = False

        self.predictor = build_sam3_predictor(
            version=self.tracker_name,
            checkpoint_path=checkpoint_path,
            compile=compile_model,
            warm_up=warm_up,
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            use_rope_real=use_rope_real,
            async_loading_frames=async_loading_frames,
        )

    @staticmethod
    def _normalize_tracker_name(tracker_name):
        name = (tracker_name or "sam3.1").lower().replace("_", ".")
        if name in {"sam31", "sam3.1", "sam3-1"}:
            return "sam3.1"
        if name == "sam3":
            return "sam3"
        raise ValueError("tracker_name must be 'sam3' or 'sam3.1'")

    @torch.inference_mode()
    def initialize(self, resource_path, bbox, image_size=None):
        """Start a session on a video/frame folder and add a frame-0 box prompt.

        Args:
            resource_path: Folder of frames or a video path accepted by SAM3 sessions.
            bbox: Absolute [x, y, w, h] initialization box.
            image_size: Optional (width, height). If omitted, inferred from resource_path.
        """
        self.close()
        self.frame_outputs = {}
        self.target_obj_id = None
        self._forward_propagated = False
        self.init_bbox = [float(v) for v in bbox]

        if image_size is None:
            image_size = self._infer_image_size(resource_path)
        self.img_width, self.img_height = image_size

        response = self.predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(resource_path),
            }
        )
        self.session_id = response["session_id"]

        rel_box = self._absolute_xywh_to_relative(self.init_bbox)
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": self.session_id,
                "frame_index": 0,
                "bounding_boxes": rel_box,
                "bounding_box_labels": np.array([1], dtype=np.int32),
                "output_prob_thresh": self.output_prob_thresh,
            }
        )
        frame_idx = int(response["frame_index"])
        outputs = response["outputs"]
        self.frame_outputs[frame_idx] = outputs
        self.target_obj_id = self._first_obj_id(outputs)

        result = self._outputs_to_result(outputs)
        if result["pred_bbox"] is None:
            result["pred_bbox"] = self.init_bbox
        return result

    @torch.inference_mode()
    def track(self, frame_idx):
        """Return prediction for one frame from the active session."""
        if self.session_id is None:
            raise RuntimeError("Tracker is not initialized. Call initialize first.")

        frame_idx = int(frame_idx)
        if frame_idx not in self.frame_outputs and not self._forward_propagated:
            self._propagate_forward()

        outputs = self.frame_outputs.get(frame_idx)

        if outputs is None:
            return {"pred_mask": None, "pred_bbox": None, "obj_id": None}
        return self._outputs_to_result(outputs)

    def _propagate_forward(self):
        for response in self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "output_prob_thresh": self.output_prob_thresh,
            }
        ):
            out_frame_idx = int(response["frame_index"])
            self.frame_outputs[out_frame_idx] = response["outputs"]
        self._forward_propagated = True

    @torch.inference_mode()
    def close(self):
        if self.session_id is None:
            return
        self.predictor.handle_request(
            {
                "type": "close_session",
                "session_id": self.session_id,
            }
        )
        self.session_id = None

    def shutdown(self):
        self.close()
        if hasattr(self.predictor, "shutdown"):
            self.predictor.shutdown()

    def _outputs_to_result(self, outputs):
        obj_ids = np.asarray(outputs.get("out_obj_ids", []))
        obj_idx = self._select_obj_index(obj_ids)
        if obj_idx is None:
            return {"pred_mask": None, "pred_bbox": None, "obj_id": None}

        if self.target_obj_id is None:
            self.target_obj_id = int(obj_ids[obj_idx])

        masks = outputs.get("out_binary_masks", None)
        pred_mask = None
        if masks is not None:
            masks = np.asarray(masks)
            if masks.ndim >= 3 and obj_idx < masks.shape[0]:
                pred_mask = masks[obj_idx].astype(np.uint8)

        pred_bbox = mask2box(pred_mask)
        if pred_bbox is None:
            boxes = np.asarray(outputs.get("out_boxes_xywh", []), dtype=np.float32)
            if boxes.ndim == 2 and obj_idx < boxes.shape[0]:
                pred_bbox = self._relative_xywh_to_absolute(boxes[obj_idx])

        return {
            "pred_mask": pred_mask,
            "pred_bbox": pred_bbox,
            "obj_id": int(obj_ids[obj_idx]),
        }

    def _select_obj_index(self, obj_ids):
        if obj_ids.size == 0:
            return None
        if self.target_obj_id is not None:
            matches = np.where(obj_ids == self.target_obj_id)[0]
            if matches.size:
                return int(matches[0])
            return None
        return 0

    @staticmethod
    def _first_obj_id(outputs):
        obj_ids = np.asarray(outputs.get("out_obj_ids", []))
        if obj_ids.size == 0:
            return None
        return int(obj_ids[0])

    def _absolute_xywh_to_relative(self, bbox):
        x, y, w, h = [float(v) for v in bbox]
        return np.array(
            [[x / self.img_width, y / self.img_height, w / self.img_width, h / self.img_height]],
            dtype=np.float32,
        )

    def _relative_xywh_to_absolute(self, bbox):
        x, y, w, h = [float(v) for v in bbox]
        return [
            x * self.img_width,
            y * self.img_height,
            w * self.img_width,
            h * self.img_height,
        ]

    @staticmethod
    def _infer_image_size(resource_path):
        path = Path(resource_path)
        if path.is_dir():
            frame_paths = []
            for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
                frame_paths.extend(path.glob(pattern))
            if not frame_paths:
                raise RuntimeError(f"No image frames found in {resource_path}")
            frame_path = sorted(frame_paths)[0]
        else:
            frame_path = path

        with Image.open(frame_path) as image:
            return image.size
