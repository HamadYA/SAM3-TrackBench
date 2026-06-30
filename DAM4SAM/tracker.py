from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from unified.config import load_config

import os
import random
import numpy as np
import yaml
import torch
import torchvision.transforms.functional as F
from sam3.model_builder import build_sam3_video_model
from collections import OrderedDict
from utils.utils import keep_largest_component, determine_tracker

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType

config = load_config(__file__)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

class SAMTracker():
    def __init__(self, tracker_name="sam3"):
        
        # Image preprocessing parameters
        self.tracker_name = tracker_name
        self.tracking_times = []
        self._needs_preflight = False

        sam3_model = build_sam3_video_model(apply_temporal_disambiguation=False)
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        self.predictor = predictor

        self.input_image_size = 1008
        self.input_image_size = getattr(self.predictor, "image_size", self.input_image_size)

        # ImageNet
        # self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        # self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        self.img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[:, None, None]
        
    def _prepare_image(self, img_pil):
        # _load_img_as_tensor from SAM2
        device = self.inference_state.get("storage_device", self.inference_state["device"])
        img = torch.from_numpy(np.array(img_pil.convert("RGB"))).to(device)
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean.to(device)) / self.img_std.to(device)     
        return img


    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        """
        Initialize the tracker with the first frame and mask.
        Function builds the SAM (2 or 2.1) tracker and initializes it with the first frame and mask.

        Args:
        - image (PIL Image): First frame of the video.
        - init_mask (numpy array): Binary mask for the initialization
        
        Returns:
        - out_dict (dict): Dictionary containing the mask for the initialization frame.
        """
        if type(init_mask) is list:
            init_mask = init_mask[0]
        
        self.frame_index = 0 # Current frame index, updated frame-by-frame
        self.object_sizes = [] # List to store object sizes (number of pixels) 
        self.last_added = -1 # Frame index of the last added frame into DRM memory
        self.img_width = image.width # Original image width
        self.img_height = image.height # Original image height

        # Build the inference state expected by SAM3 tracker
        self.inference_state = self.predictor.init_state(
            video_height=self.img_height,
            video_width=self.img_width,
            num_frames=1,
        )
        # SAM3 init_state does not allocate images when video_path is None
        if "images" not in self.inference_state:
            self.inference_state["images"] = {}
        prepared_img = self._prepare_image(image)
        self.inference_state["images"][0] = prepared_img
        self.inference_state["num_frames"] = 1
        self._needs_preflight = True

        if init_mask is None:
            if bbox is None:
                raise ValueError("Error: initialization state (bbox or mask) is not given.")
            # Use the box prompt to obtain the initial mask
            rel_box = np.array(
                [
                    [
                        bbox[0] / self.img_width,
                        bbox[1] / self.img_height,
                        (bbox[0] + bbox[2]) / self.img_width,
                        (bbox[1] + bbox[3]) / self.img_height,
                    ]
                ],
                dtype=np.float32,
            )
            _, _, _, video_res_masks = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=0,
                box=rel_box,
            )
        else:
            mask_tensor = torch.as_tensor(init_mask, dtype=torch.float32)
            _, _, _, video_res_masks = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=0,
                mask=mask_tensor,
            )

        m = (video_res_masks[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
        return {"pred_mask": m}

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        Function to track the object in the next frame.

        Args:
        - image (PIL Image): Next frame of the video.
        - init (bool): Whether the current frame is the initialization frame.

        Returns:
        - out_dict (dict): Dictionary containing the predicted mask for the current frame.
        """
        # torch.cuda.empty_cache()
        # Prepare the image for input to the model
        prepared_img = self._prepare_image(image)
        if not init:
            self.frame_index += 1
        # Update frame bookkeeping for streaming usage
        self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        out_mask = None
        run_preflight = self._needs_preflight
        # Propagate the tracking to the next frame
        # return_all_masks=True returns all predicted (chosen and alternative) masks and corresponding IoUs
        for out in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.frame_index, max_frame_num_to_track=0, return_all_masks=True, reverse=False, propagate_preflight=run_preflight):
            if len(out) == 5:
                # There are 5 outputs when the tracking is done on the initialization frame
                out_frame_idx, _, low_res_masks, video_res_masks, obj_scores = out
                alternative_masks_ious = None
                m = (video_res_masks[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
                out_mask = m
            else:
                # There are 6 outputs when the tracking is done on a non-initialization frame
                # alternative_masks_ious is a tuple containing chosen and alternative masks and corresponding predicted IoUs
                out_frame_idx, _, low_res_masks, video_res_masks, obj_scores, alternative_masks_ious = out
                m = (video_res_masks[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
                out_mask = m

                alternative_masks, out_all_ious = alternative_masks_ious # Extract all predicted masks (chosen and alternatives) and IoUs
                m_idx = np.argmax(out_all_ious) # Index of the chosen predicted mask
                m_iou = out_all_ious[m_idx] # Predicted IoU of the chosen predicted mask
                # Delete the chosen predicted mask from the list of all predicted masks, leading to only alternative masks
                alternative_masks = [mask for i, mask in enumerate(alternative_masks) if i != m_idx]

                # Determine if the object ratio between the current frame and the previous frames is within a certain range
                n_pixels = (m == 1).sum() 
                self.object_sizes.append(n_pixels)
                if len(self.object_sizes) > 1 and n_pixels >= 1:
                    obj_sizes_ratio = n_pixels / np.median([
                        size for size in self.object_sizes[-300:] if size >= 1
                    ][-10:])
                else:
                    obj_sizes_ratio = -1

                # The first condition checks if:
                #  - the chosen predicted mask has a high predicted IoU, 
                #  - the object size ratio is within a +- 20% range compared to the previous frames, 
                #  - the target is present in the current frame,
                #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
                if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and n_pixels >= 1 and (self.frame_index - self.last_added > 5 or self.last_added == -1):
                    alternative_masks = [Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8) 
                                    for m_ in alternative_masks]

                    # Numpy array of the chosen mask and corresponding bounding box
                    chosen_mask_np = m.copy()
                    chosen_bbox = Mask(chosen_mask_np).convert(RegionType.RECTANGLE)

                    # Delete the parts of the alternative masks that overlap with the chosen mask
                    alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                    # Keep only the largest connected component of the processed alternative masks
                    alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                    if len(alternative_masks) > 0:
                        # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                        alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                        # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                        alternative_bboxes = [Mask(m_).convert(RegionType.RECTANGLE) for m_ in alternative_masks]
                        # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                        ious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]
                        
                        # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                        # That would mean that there are significant differences between the chosen mask and the processed alternative masks, 
                        # leading to possible detections of distractors within alternative masks.
                        if np.min(np.array(ious)) <= 0.7:
                            self.last_added = self.frame_index # Update the last added frame index
                            self.predictor.add_to_drm(
                                inference_state=self.inference_state,
                                frame_idx=out_frame_idx,
                                obj_id=0,
                            )

        # Preflight is only needed once, after receiving the first prompt
        self._needs_preflight = False
        # Release frames from the buffer once they have been processed
        self.inference_state["images"].pop(self.frame_index, None)
        if run_preflight:
            self.inference_state["images"].pop(0, None)

        return {"pred_mask": out_mask}

    def estimate_mask_from_box(self, bbox):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._get_image_feature(self.inference_state, 0, 1)

        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None, :]
        box = torch.as_tensor(box, dtype=torch.float, device=current_vision_feats[0].device)

        from sam3.model.utils.sam1_utils import SAM2Transforms
        _transforms = SAM2Transforms(
            resolution=self.predictor.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        unnorm_box = _transforms.transform_boxes(
            box, normalize=True, orig_hw=(self.img_height, self.img_width)
        )  # Bx2x2
        
        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
        box_labels = box_labels.repeat(unnorm_box.size(0), 1)
        concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(current_vision_feats[i].permute(1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))
        if self.predictor.directly_add_no_mem_embed:
            img_embed = current_vision_feats[2] + self.predictor.no_mem_embed
        else:
            img_embed = current_vision_feats[2]
        _, b_, c_ = current_vision_feats[2].shape
        img_embed = img_embed.permute(1, 2, 0).view(b_, c_, feat_sizes[2][0], feat_sizes[2][1])
        low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = _transforms.postprocess_masks(
            low_res_masks, (self.img_height, self.img_width)
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        masks = masks > 0

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()

        init_mask = masks_np[0]
        return init_mask
    