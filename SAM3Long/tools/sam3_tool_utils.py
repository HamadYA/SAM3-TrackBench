import inspect
from contextlib import nullcontext

import numpy as np
import torch


def autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_sam3_tracker_predictor(
    checkpoint_path=None,
    load_from_hf=True,
    compile_model=False,
    apply_temporal_disambiguation=True,
):
    """Build the SAM3 tracker used by the video examples."""
    from sam3.model_builder import build_sam3_video_model

    sam3_model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        compile=compile_model,
        apply_temporal_disambiguation=apply_temporal_disambiguation,
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    return predictor


def set_sam3_memory_selection_params(
    inference_state,
    num_pathway=3,
    iou_thre=0.1,
    uncertainty=2,
):
    inference_state["num_pathway"] = num_pathway
    inference_state["iou_thre"] = iou_thre
    inference_state["uncertainty"] = uncertainty


def add_sam3_mask_prompt(predictor, inference_state, frame_idx, obj_id, mask):
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.float32)
    if mask.dtype == torch.bool:
        mask = mask.float()
    return predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask,
    )


def reset_sam3_tracker_state(predictor, inference_state):
    if hasattr(predictor, "reset_state"):
        predictor.reset_state(inference_state)
    elif hasattr(predictor, "clear_all_points_in_video"):
        predictor.clear_all_points_in_video(inference_state)
    else:
        raise RuntimeError("SAM3 predictor does not expose a state reset method")


def _propagate_kwargs(
    predictor,
    inference_state,
    start_frame_idx,
    max_frame_num_to_track,
    reverse,
    propagate_preflight,
):
    kwargs = {
        "inference_state": inference_state,
        "start_frame_idx": start_frame_idx,
        "max_frame_num_to_track": max_frame_num_to_track,
        "reverse": reverse,
    }
    signature = inspect.signature(predictor.propagate_in_video)
    if "propagate_preflight" in signature.parameters:
        kwargs["propagate_preflight"] = propagate_preflight
    return kwargs


def iter_sam3_propagation(
    predictor,
    inference_state,
    start_frame_idx=0,
    max_frame_num_to_track=None,
    reverse=False,
    propagate_preflight=False,
):
    """Yield ``(frame_idx, obj_ids, masks)`` for both SAM3 propagation APIs.

    The local tracker in this tree returns ``(obj_ids, list_of_masks)`` while the
    public example API yields per-frame tuples. This keeps the tools robust to
    either convention.
    """
    result = predictor.propagate_in_video(
        **_propagate_kwargs(
            predictor,
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
            propagate_preflight,
        )
    )

    if isinstance(result, tuple) and len(result) == 2:
        obj_ids, masks_per_frame = result
        for offset, masks in enumerate(masks_per_frame):
            frame_idx = start_frame_idx - offset if reverse else start_frame_idx + offset
            yield frame_idx, obj_ids, masks
        return

    for item in result:
        if isinstance(item, dict):
            frame_idx = item["frame_index"]
            outputs = item["outputs"]
            yield frame_idx, outputs.get("out_obj_ids", []), outputs["out_binary_masks"]
            continue

        if len(item) >= 5:
            frame_idx, obj_ids, _, video_res_masks, _ = item[:5]
        elif len(item) == 4:
            frame_idx, obj_ids, _, video_res_masks = item
        elif len(item) == 3:
            frame_idx, obj_ids, video_res_masks = item
        else:
            raise RuntimeError(f"Unexpected SAM3 propagation output: {type(item)!r}")
        yield frame_idx, obj_ids, video_res_masks


def score_array_for_object(obj_ids, masks, object_id):
    obj_ids = np.asarray(obj_ids).tolist()
    if object_id in obj_ids:
        obj_idx = obj_ids.index(object_id)
    elif len(obj_ids) == 1:
        obj_idx = 0
    else:
        return None

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if masks.ndim == 4:
        mask = masks[obj_idx, 0]
    elif masks.ndim == 3:
        mask = masks[obj_idx]
    elif masks.ndim == 2 and len(obj_ids) == 1:
        mask = masks
    else:
        raise RuntimeError(f"Unexpected SAM3 mask shape: {masks.shape}")
    return mask


def mask_array_for_object(obj_ids, masks, object_id, score_thresh=0.0):
    mask = score_array_for_object(obj_ids, masks, object_id)
    if mask is None:
        return None
    return mask > score_thresh
