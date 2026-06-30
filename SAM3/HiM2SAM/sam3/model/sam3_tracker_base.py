# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging

import torch
import torch.nn.functional as F

from sam3.model.memory import SimpleMaskEncoder

from sam3.model.sam3_tracker_utils import get_1d_sine_pe, select_closest_cond_frames

from sam3.sam.mask_decoder import MaskDecoder, MLP
from sam3.sam.prompt_encoder import PromptEncoder
from sam3.sam.transformer import TwoWayTransformer
from sam3.train.data.collator import BatchedDatapoint

try:
    from timm.layers import trunc_normal_
except ModuleNotFoundError:
    # compatibility for older timm versions
    from timm.models.layers import trunc_normal_

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


import time
import os
from sam3.model.utils.kalman_filter import KalmanFilter
import cv2

#test for mask selection
from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType
from scipy.spatial.distance import directed_hausdorff
import pdb

import numpy as np
import sys

def keep_largest_component(mask):
    """
    Keeps only the largest connected component from a binary mask.
    
    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.
    
    Returns:
    - filtered_mask (numpy array): Binary mask with only the largest connected component.
    """
    # Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Find the index of the largest component (excluding background)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)
    # Create a mask that contains only the largest component
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_component] = 1
    return filtered_mask

def extract_dense_boundary(mask):
    """
    Extracts dense boundary points from a binary mask using morphological gradient.

    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.

    Returns:
    - points (numpy array): An (N, 2) array of (row, col) coordinates representing boundary points.
    """
    kernel = np.ones((3,3), np.uint8)  # 3x3 结构元素
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)  # 提取边界
    points = np.column_stack(np.where(boundary > 0))  # 获取边界点坐标
    return points

def hausdorff_distance(mask1, mask2):
    """
    Computes the directed Hausdorff distance between two binary masks based on their dense boundary points.

    Args:
    - mask1 (numpy array): First 2D binary mask.
    - mask2 (numpy array): Second 2D binary mask.

    Returns:
    - distance (float): Directed Hausdorff distance from mask1 to mask2.
                        Returns 0.0 if either mask has no boundary points.
    """
    points1 = extract_dense_boundary(mask1)
    points2 = extract_dense_boundary(mask2)
    if len(points1) == 0 or len(points2) == 0:
        return float(0)

    return directed_hausdorff(points2, points1)[0]


class Sam3TrackerBase(torch.nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        maskmem_backbone,
        num_maskmem=7,  # default 1 input frame + 6 previous frames as in CAE
        image_size=1008,
        backbone_stride=14,  # stride of the image backbone output
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # Whether to always keep the first conditioning frame in case we exceed the maximum number of conditioning frames allowed
        keep_first_cond_frame=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to offload outputs to CPU memory during evaluation, to avoid GPU OOM on very long videos or very large resolutions or too many objects
        # (it's recommended to use `forward_backbone_per_frame_for_eval=True` first before setting this option to True)
        offload_output_to_cpu_for_eval=False,
        # whether to trim the output of past non-conditioning frames (num_maskmem frames before the current frame) during evaluation
        # (this helps save GPU or CPU memory on very long videos for semi-supervised VOS eval, where only the first frame receives prompts)
        trim_past_non_cond_mem_for_eval=False,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # the maximum number of object pointers from other frames in encoder cross attention
        max_obj_ptrs_in_encoder=16,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        # whether to compile all the model compoents
        compile_all_components=False,
        # select the frame with object existence
        use_memory_selection=False,
        # when using memory selection, the threshold to determine if the frame is good
        mf_threshold=0.01,

        # ==============================================
        # Whether to use kf_mode or original SAM 2
        kf_mode: bool = False,
        # Hyperparameters for kalman filter
        stable_frames_threshold: int = 15,
        stable_ious_threshold: float = 0.3,
        min_obj_score_logits: float = -1,
        kf_score_weight: float = 0.15,
        memory_bank_iou_threshold: float = 0.5,
        memory_bank_obj_score_threshold: float = 0.0,
        memory_bank_kf_score_threshold: float = 0.0,
        # reversecot
        rvcot_mode=True,
        rvcot_weight=0.25,
        rvcot_ious_threshold=0.8,
        memory_bank_rvcot_iou_threshold = 0.5,
        rvcot_frame_far = 5,
        rvcot_iou_aggregation_method = 'farest',
        rvcot_mirrow_padding = False,
        rvcot_mem_selection=True,
        rvcot_mem_sel_intv=5,
        rvcot_mem_sel_short_intv=2,
        rvcot_mem_long_len=1,
        sample_count=100,
        #dam4sam iou threshold
        rvcot_mem_sel_intioutred = 0.7,
        rvcot_mem_selection_method = 'hsdf_dist',
        #hsdf_dist measure
        rvcot_siou_ratio_threshold = 1.5,
        rvcot_mem_sel_hsf_thred_ratio = 2.0,
        rvcot_mem_sel_motionfilter_flag = True,
        rcvot_idxreverse = False,
        rvcot_memious_threshold=None,
        rvcot_predict_iouweight=0.7
        # ==============================================
    ):
        super().__init__()

        # Part 1: the image backbone
        self.backbone = backbone
        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        # A conv layer to downsample the GT mask prompt to stride 4 (the same stride as
        # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
        # so that it can be fed into the SAM mask decoder to generate a pointer.
        self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)

        # Part 2: encoder-only transformer to fuse current frame's visual features
        # with memories from past frames
        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible

        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)

        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        # we resize the mask if it doesn't match `self.input_mask_size` (which is always 4x
        # the low-res mask size, regardless of the actual input image size); this is because
        # `_use_mask_as_output` always downsamples the input masks by 4x
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
        trunc_normal_(self.no_obj_ptr, std=0.02)
        self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame

        # Use frame filtering according to SAM2Long
        self.use_memory_selection = use_memory_selection
        self.mf_threshold = mf_threshold



        # Whether to use kalman_filter or original SAM 2
        self.kf_mode = kf_mode

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0

        # Debug purpose
        self.history = {} # debug
        self.frame_cnt = 0 # debug

        # Hyperparameters for KF
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold

        # reversecot
        self.rvcot_mode = rvcot_mode
        self.rvcot_weight = rvcot_weight
        self.rvcot_ious_threshold = rvcot_ious_threshold
        self.rvcot_frame_far=rvcot_frame_far
        self.rvcot_iou_aggregation_method = rvcot_iou_aggregation_method
        self.rvcot_rvcot_mirrow_padding  = rvcot_mirrow_padding
        self.rvcot_mem_selection = rvcot_mem_selection
        self.rvcot_mem_selection_highconf_frameidx = []
        self.rvcor_area_list = []
        self.rvcot_mem_sel_intv = rvcot_mem_sel_intv
        self.rvcot_mem_long_len =rvcot_mem_long_len
        self.memory_bank_rvcot_iou_threshold = memory_bank_rvcot_iou_threshold
        self.rvcot_mem_sel_short_intv = rvcot_mem_sel_short_intv
        self.rvcot_mem_sel_intioutred = rvcot_mem_sel_intioutred
        self.rvcot_mem_selection_method = rvcot_mem_selection_method
        self.rvcot_siou_ratio_threshold = rvcot_siou_ratio_threshold
        self.rvcot_mem_sel_hsf_thred_ratio = rvcot_mem_sel_hsf_thred_ratio
        self.rvcot_mem_sel_motionfilter_flag = rvcot_mem_sel_motionfilter_flag
        self.sample_count = sample_count
        self.rvcot_filter = None
        self.rcvot_idxreverse = rcvot_idxreverse
        self.rvcot_memious_threshold = rvcot_ious_threshold if rvcot_memious_threshold is None else rvcot_memious_threshold
        self.rvcot_predict_iouweight = rvcot_predict_iouweight
        if rvcot_mode:
            assert(not self.kf_mode),print("!Conflict aux mode!")
            from sam3.model.utils.cot_filter import RVCotFilter
            self.rvcot_filter = RVCotFilter(
                checkpoint=None,box_iou=True,
                frame_far=self.rvcot_frame_far,
                mirrow_padding=self.rvcot_rvcot_mirrow_padding,
                iouweight=self.rvcot_predict_iouweight)

        print(f"\033[93mKF mode: {self.kf_mode}\033[0m")
        print(f"\033[93mRVCOT mode: {self.rvcot_mode} with KF TRUE \033[0m")


        # Compile all components of the model
        self.compile_all_components = compile_all_components
        if self.compile_all_components:
            self._compile_all_components()

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None, dummy=False):
        if dummy:
            return torch.zeros(len(rel_pos_list), self.mem_dim, device=device)

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = (
            torch.tensor(rel_pos_list).pin_memory().to(device=device, non_blocking=True)
            / t_diff_max
        )
        tpos_dim = self.hidden_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        pos_enc = self.obj_ptr_tpos_proj(pos_enc)

        return pos_enc

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        # a linear projection on SAM output tokens to turn them into object pointers
        self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        # a linear projection on temporal positional encoding in object pointers to
        # avoid potential interference with spatial positional encoding
        self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        gt_masks=None,
        # =================================
        inference_state=None,
        frame_idx=None
        # =================================
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        # Clone image_pe and the outputs of sam_prompt_encoder
        # to enable compilation
        sparse_embeddings = self._maybe_clone(sparse_embeddings)
        dense_embeddings = self._maybe_clone(dense_embeddings)
        image_pe = self._maybe_clone(self.sam_prompt_encoder.get_dense_pe())
        with torch.profiler.record_function("sam_mask_decoder"):
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=False,  # the image is already batched
                high_res_features=high_res_features,
            )
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)
        object_score_logits = self._maybe_clone(object_score_logits)

        if self.training and self.teacher_force_obj_scores_for_mem:
            # we use gt to detect if there is an object or not to
            # select no obj ptr and use an empty mask for spatial memory
            is_obj_appearing = torch.any(gt_masks.float().flatten(1) > 0, dim=1)
            is_obj_appearing = is_obj_appearing[..., None]
        else:
            # is_obj_appearing = object_score_logits > 0
            is_obj_appearing = object_score_logits > self.min_obj_score_logits

        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
        # consistent with the actual mask prediction
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        kf_ious = None
        rvcot_ious = None
        ### KF mode
        if multimask_output and self.kf_mode:
            if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                #valid mask
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                #initiate kf
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
                self.stable_frames += 1
            # not enough init frames
            elif self.stable_frames < self.stable_frames_threshold:
                # t+1 KF predict
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                if ious[0][best_iou_inds] > self.stable_ious_threshold:
                    self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
            # valid KF 
            else:
                # KF predict step
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                high_res_multibboxes = []
                batch_inds = torch.arange(B, device=device)
                for i in range(ious.shape[1]):
                    non_zero_indices = torch.argwhere(high_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_multibboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
                # compute the IoU between the predicted bbox and the high_res_multibboxes
                kf_ious = torch.tensor(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes), device=device)
                weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
                best_iou_inds = torch.argmax(weighted_ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
                if False:
                    # make all these on cpu                        
                    self.history[self.frame_cnt] = {
                        "kf_predicted_bbox": self.kf.xyah_to_xyxy(self.kf_mean[:4]),
                        # "multi_masks": high_res_multimasks.cpu(),
                        "ious": ious.cpu(),
                        "multi_bboxes": high_res_multibboxes,
                        "kf_ious": kf_ious,
                        "weighted_ious": weighted_ious.cpu(),
                        "final_selection": best_iou_inds.cpu(),
                    }
                self.frame_cnt += 1
                if ious[0][best_iou_inds] < self.stable_ious_threshold:
                    self.stable_frames = 0
                else:
                    self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds]))
                
        elif multimask_output and self.rvcot_mode and inference_state is not None:
            # KF strategy
            if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                #initiate kf
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
                self.stable_frames += 1
            elif self.stable_frames < self.stable_frames_threshold:
                # t+1 KF predict
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                best_iou_inds = torch.argmax(ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                if ious[0][best_iou_inds] > self.stable_ious_threshold:
                    self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
                if ious[0][best_iou_inds] < self.rvcot_ious_threshold:
                    # using point tracker when KF not reliable
                    rvcot_ious = self.rvcot_filter.predict(frame_idx, inference_state,high_res_multimasks,
                                                           iou_aggregation_method = self.rvcot_iou_aggregation_method,
                                                           sample_count=self.sample_count)
                    print('rvcot',frame_idx)
                    rvcot_weighted_ious = self.rvcot_weight * rvcot_ious + (1-self.rvcot_weight)* ious
                    best_iou_inds = torch.argmax(rvcot_weighted_ious, dim=-1)
                    batch_inds = torch.arange(B, device=device)
                    low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                    high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                    non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                self.frame_cnt += 1
            else:
                # KF predict step
                self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                high_res_multibboxes = []
                batch_inds = torch.arange(B, device=device)
                for i in range(ious.shape[1]):
                    non_zero_indices = torch.argwhere(high_res_multimasks[batch_inds, i].unsqueeze(1)[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_multibboxes.append([0, 0, 0, 0])
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_multibboxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
                # compute the IoU between the predicted bbox and the high_res_multibboxes
                kf_ious = torch.tensor(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes), device=device)
                weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
                best_iou_inds = torch.argmax(weighted_ious, dim=-1)
                if ious[0][best_iou_inds] < self.stable_ious_threshold or weighted_ious[0][best_iou_inds]< self.rvcot_ious_threshold:
                    rvcot_ious = self.rvcot_filter.predict(frame_idx, inference_state,high_res_multimasks,
                                                           iou_aggregation_method = self.rvcot_iou_aggregation_method,
                                                           sample_count = self.sample_count)
                    print('rvcot',frame_idx)
                    rvcot_weighted_ious = self.rvcot_weight * rvcot_ious + (1-self.rvcot_weight)* weighted_ious
                    best_iou_inds = torch.argmax(rvcot_weighted_ious, dim=-1)
                batch_inds = torch.arange(B, device=device)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
                
                self.frame_cnt += 1
                ###update KF 
                if ious[0][best_iou_inds] < self.stable_ious_threshold:
                    self.stable_frames = 0
                else:
                    self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds]))
            # memory selection
            if self.rvcot_mem_selection and self.rvcot_mem_selection_method == 'default':
                cur_area = (high_res_masks>0).sum().cpu().numpy()
                self.rvcor_area_list.append(cur_area)
                if len(self.rvcor_area_list) > 1 and cur_area >= 1:
                    obj_sizes_ratio = cur_area / np.median([
                            size for size in self.rvcor_area_list[-300:] if size >= 1
                        ][-10:])
                else:
                    obj_sizes_ratio = -1

                siou = ious[0][best_iou_inds][0]
                kiou = kf_ious[best_iou_inds][0] if kf_ious is not None else None
                iouflag = siou > 0.8
                if iouflag and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and cur_area >= 1 :
                    # other mask proposals
                    alternative_masks =  [high_res_multimasks[0, i].cpu().numpy() for i in range(3) if i!=best_iou_inds]
                                     
                    # Numpy array of the chosen mask and corresponding bounding box
                    chosen_mask_np = (high_res_multimasks[0, best_iou_inds[0]]>0).cpu().numpy().astype(np.uint8).copy()
                    chosen_bbox = Mask(chosen_mask_np>0).convert(RegionType.RECTANGLE)

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
                        memious = [calculate_overlaps([chosen_bbox], [bbox])[0] for bbox in alternative_bboxes]
                        
                        # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                        # That would mean that there are significant differences between the chosen mask and the processed alternative masks, 
                        # leading to possible detections of distractors within alternative masks.
                        # [iou - far away from target -> distractor]
                        if np.min(np.array(memious)) <= self.rvcot_mem_sel_intioutred and frame_idx>0:
                            if len(self.rvcot_mem_selection_highconf_frameidx)==0:
                                lst = -self.rvcot_mem_sel_intv-1
                            else:
                                lst = self.rvcot_mem_selection_highconf_frameidx[-1]
                            if lst + self.rvcot_mem_sel_intv <= frame_idx:
                                self.rvcot_mem_selection_highconf_frameidx.append(frame_idx)# Update the last added frame index
            elif self.rvcot_mem_selection and self.rvcot_mem_selection_method == 'hsdf_dist':
                ### area constraint
                cur_area = (high_res_masks>0).sum().cpu().numpy()
                cur_d = 2*np.sqrt(cur_area /np.pi) 
                self.rvcor_area_list.append(cur_area)
                if len(self.rvcor_area_list) > 1 and cur_area >= 1:
                    obj_sizes_ratio = cur_area / np.median([
                            size for size in self.rvcor_area_list[-300:] if size >= 1
                        ][-10:])
                else:
                    obj_sizes_ratio = -1
                ### iou_score ratio constraint
                siou = ious[0][best_iou_inds]
                kiou = kf_ious[best_iou_inds][0] if kf_ious is not None else None
                iouflag = siou > 0.8
                if kiou is not None and self.rvcot_mem_sel_motionfilter_flag:
                    iouflag = iouflag  and kiou > self.rvcot_ious_threshold
                iou_list = [ious[0][i] for i in range(len(ious[0]))]
                iou_list.sort(reverse=True)
                iou_ratio = iou_list[0] / iou_list[1]
                siou_ratio_flag = (iou_ratio > self.rvcot_siou_ratio_threshold)
                if siou_ratio_flag and obj_sizes_ratio >= 0.75 and obj_sizes_ratio <= 1.25 and iouflag and  cur_area >= 1 :
                    # hsdf dist constraint
                    chosen_mask_np = (high_res_multimasks[0, best_iou_inds[0]]>0).cpu().numpy().astype(np.uint8).copy()
                    alternative_masks =  [(high_res_multimasks[0, i]>0).cpu().numpy().astype(np.uint8) for i in range(3) if i!=best_iou_inds]
                    alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8)  for m_ in alternative_masks]
                    #discard noise using largest component
                    alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]

                    if len(alternative_masks) > 0:
                        hsf_dit = [hausdorff_distance(chosen_mask_np, m_) for m_ in alternative_masks]
                        if np.max(np.array(hsf_dit)) > self.rvcot_mem_sel_hsf_thred_ratio * cur_d and frame_idx>0:
                            if len(self.rvcot_mem_selection_highconf_frameidx)==0:
                                lst = -self.rvcot_mem_sel_intv-1
                            else:
                                lst = self.rvcot_mem_selection_highconf_frameidx[-1]
                            if lst + self.rvcot_mem_sel_intv <= frame_idx:
                                self.rvcot_mem_selection_highconf_frameidx.append(frame_idx)# Update the last added frame index         
        elif multimask_output and self.rvcot_filter is not None:
            # RVCOT is enabled but we still need to select a single mask for memory encoding
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        elif multimask_output and not self.kf_mode and not self.rvcot_filter:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            best_iou_inds = 0
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()

        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,

            ious[0][best_iou_inds],
            kf_ious[best_iou_inds] if kf_ious is not None else None,
            rvcot_ious[best_iou_inds] if rvcot_ious is not None else None
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(
                high_res_masks.size(-2) // self.backbone_stride * 4,
                high_res_masks.size(-1) // self.backbone_stride * 4,
            ),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1, 1).float()
        # produce an object pointer using the SAM decoder from the mask input
        _, _, _, _, _, obj_ptr, _, _, _,_ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            None,None,None,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            ious,
            None,
            None,
        )

    def forward(self, input: BatchedDatapoint, is_inference=False):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM3VideoPredictor for inference."
            "See examples/sam3_dense_video_tracking.ipynb for an inference example."
        )

    def forward_image(self, img_batch):
        """Get the image feature on the input batch."""
        # This line is the only change from the parent class
        # to use the SAM3 backbone instead of the SAM2 backbone.
        backbone_out = self.backbone.forward_image(img_batch)["sam2_backbone_out"]
        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        # Clone to help torch.compile
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = self._maybe_clone(
                backbone_out["backbone_fpn"][i]
            )
            backbone_out["vision_pos_enc"][i] = self._maybe_clone(
                backbone_out["vision_pos_enc"][i]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features (same as in MDETR_API model)."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repeatitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def cal_mem_score(self, object_score_logits, iou_score):
        object_score_norm = torch.where(
            object_score_logits > 0,
            object_score_logits.sigmoid() * 2 - 1,  ## rescale to [0, 1]
            torch.zeros_like(object_score_logits),
        )
        score_per_frame = (object_score_norm * iou_score).mean()
        return score_per_frame

    def frame_filter(self, output_dict, track_in_reverse, frame_idx, num_frames, r):
        if (frame_idx == 0 and not track_in_reverse) or (
            frame_idx == num_frames - 1 and track_in_reverse
        ):
            return []

        max_num = min(
            num_frames, self.max_obj_ptrs_in_encoder
        )  ## maximum number of pointer memory frames to consider

        if not track_in_reverse:
            start = frame_idx - 1
            end = 0
            step = -r
            must_include = frame_idx - 1
        else:
            start = frame_idx + 1
            end = num_frames
            step = r
            must_include = frame_idx + 1

        valid_indices = []
        for i in range(start, end, step):
            if (
                i not in output_dict["non_cond_frame_outputs"]
                or "eff_iou_score" not in output_dict["non_cond_frame_outputs"][i]
            ):
                continue

            score_per_frame = output_dict["non_cond_frame_outputs"][i]["eff_iou_score"]

            if score_per_frame > self.mf_threshold:  # threshold
                valid_indices.insert(0, i)

            if len(valid_indices) >= max_num - 1:
                break

        if must_include not in valid_indices:
            valid_indices.append(must_include)

        return valid_indices

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        use_prev_mem_frame=True,
        inference_state=None, # tracking in reverse time order (for demo usage),
        freq_output_dict=None,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame and use_prev_mem_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )
            t_pos_and_prevs = [
                ((frame_idx - t) * tpos_sign_mul, out, True)
                for t, out in selected_cond_outputs.items()
            ]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )

            if self.kf_mode:
                valid_indices = [] 
                if frame_idx > 2:  # Ensure we have previous frames to evaluate
                    for i in range(frame_idx - 2, 1, -1):  # Iterate backwards through all previous frames
                        # non-conditional frame
                        iou_score = output_dict["non_cond_frame_outputs"][i]["best_iou_score"]  # Get mask affinity score
                        obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"]  # Get object score
                        kf_score = output_dict["non_cond_frame_outputs"][i]["kf_score"] if "kf_score" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available
                        # Check if the scores meet the criteria for being a valid index
                        if iou_score.item() > self.memory_bank_iou_threshold and \
                           obj_score.item() > self.memory_bank_obj_score_threshold and \
                           (kf_score is None or kf_score.item() > self.memory_bank_kf_score_threshold):
                            valid_indices.insert(0, i)  
                        # Check the number of valid indices
                        if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:  
                            break
                if frame_idx - 1 not in valid_indices: 
                    valid_indices.append(frame_idx - 1)
                for t_pos in range(1, self.num_maskmem):  # Iterate over the number of mask memories
                    idx = t_pos - self.num_maskmem  # Calculate the index for valid indices  || 倒数valid_indices序列
                    if idx < -len(valid_indices):  # Skip if index is out of bounds
                        continue
                    out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)  # Get output for the valid index
                    if out is None:  # If not found, check unselected outputs
                        out = unselected_cond_outputs.get(valid_indices[idx], None)
                    t_pos_and_prevs.append((t_pos, out, False))  # Append the temporal position and output to the list # t_pos = 1 2 3...num_maskmem-1
            elif self.rvcot_mode and not self.rvcot_mem_selection:
              
                valid_indices = [] 
                if frame_idx > 2:  # Ensure we have previous frames to evaluate
                    for i in range(frame_idx - 2, 1, -1):  # Iterate backwards through all previous frames
                        iou_score = output_dict["non_cond_frame_outputs"][i]["best_iou_score"]  # Get mask affinity score
                        obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"]  # Get object score
                        kf_score = output_dict["non_cond_frame_outputs"][i]["kf_score"] if "kf_score" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available
                        rvcot_ious = output_dict["non_cond_frame_outputs"][i]["rvcot_ious"] if "rvcot_ious" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available

                        # Check if the scores meet the criteria for being a valid index
                        if iou_score.item() > self.memory_bank_iou_threshold and \
                           obj_score.item() > self.memory_bank_obj_score_threshold and \
                           (kf_score is None or kf_score.item() > self.memory_bank_kf_score_threshold) and \
                           (rvcot_ious is None or rvcot_ious.item() > self.memory_bank_rvcot_iou_threshold):
                            valid_indices.insert(0, i)  
                        # Check the number of valid indices
                        if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:  
                            break
                # preframe is not trustful 
                if frame_idx<=2 and frame_idx - 1 not in valid_indices: 
                    valid_indices.append(frame_idx - 1)
                for t_pos in range(1, self.num_maskmem):  # Iterate over the number of mask memories
                    idx = t_pos - self.num_maskmem  # Calculate the index for valid indices  || 倒数valid_indices序列
                    if idx < -len(valid_indices):  # Skip if index is out of bounds
                        continue
                    out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)  # Get output for the valid index
                    if out is None:  # If not found, check unselected outputs
                        out = unselected_cond_outputs.get(valid_indices[idx], None)
                    t_pos_and_prevs.append((t_pos, out, False))  # Append the temporal position and output to the list # t_pos = 1 2 3...num_maskmem-1

            elif self.rvcot_mode and  self.rvcot_mem_selection:
                ### long-term frame : tpos = 0
                ### short-term frame : tpos = 1 2 3

                ### prepare long-term frame
                long_list = []
                short_list = []
                longmem_len = self.rvcot_mem_long_len
                r = self.rvcot_mem_sel_short_intv
                for i in range(longmem_len):
                    if len(self.rvcot_mem_selection_highconf_frameidx)>i:
                        long_list.append(self.rvcot_mem_selection_highconf_frameidx[-1-i])

                long_list.sort()
                lst = np.inf
                i =self.num_maskmem -1 - len(long_list)
                prev_frame_idx = frame_idx 

                # prepare short-term frame
                while i > 0:   
                    prev_frame_idx-=1
                    if not prev_frame_idx < lst-r:
                        continue
                    if prev_frame_idx <1:
                        break
                    if prev_frame_idx >0 and prev_frame_idx not in long_list:  

                        frame_output = output_dict["frame_score_for_mem"].get(prev_frame_idx) 
                        iou_score = frame_output["best_iou_score"]  # Get mask affinity score
                        obj_score = frame_output["object_score_logits"]  # Get object score
                        kf_score = frame_output["kf_score"] if "kf_score" in frame_output else None  # Get motion score if available
                        rvcot_ious = frame_output["rvcot_ious"] if "rvcot_ious" in frame_output else None  # Get motion score if available

                        # Check if the scores meet the criteria for being a valid index
                        if iou_score > self.memory_bank_iou_threshold and \
                        obj_score > self.memory_bank_obj_score_threshold and \
                        (kf_score is None or kf_score > self.memory_bank_kf_score_threshold) and \
                        (rvcot_ious is None or rvcot_ious > self.memory_bank_rvcot_iou_threshold):
                            # if not self.rvcot_inteveal_intlike:
                            short_list.append(prev_frame_idx)      
                            lst = prev_frame_idx 
                            i-=1   
                short_list.sort()
                for frame_idx in long_list:  # Iterate over the number of mask memories
                    out = output_dict["non_cond_frame_outputs"].get(frame_idx, None)  # Get output for the valid index
                    if out is None:  # If not found, check unselected outputs
                        out = unselected_cond_outputs.get(valid_indices[idx], None)
                    t_pos_and_prevs.append((0, out, False))
                for t_pos, frame_idx in enumerate(short_list):
                    out = output_dict["non_cond_frame_outputs"].get(frame_idx, None)  # Get output for the valid index
                    if out is None:  # If not found, check unselected outputs
                        out = unselected_cond_outputs.get(valid_indices[idx], None)
                    t_pos_and_prevs.insert(1,(self.num_maskmem-t_pos-1, out, False))
            # modification end
            else: #Original sam
                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                    if t_rel == 1:
                        # for t_rel == 1, we take the last frame (regardless of r)
                        if not track_in_reverse:
                            # the frame immediately before this frame (i.e. frame_idx - 1)
                            prev_frame_idx = frame_idx - t_rel
                        else:
                            # the frame immediately after this frame (i.e. frame_idx + 1)
                            prev_frame_idx = frame_idx + t_rel
                    else:
                        # for t_rel >= 2, we take the memory frame from every r-th frames
                        if not track_in_reverse:
                            # first find the nearest frame among every r-th frames before this frame
                            # for r=1, this would be (frame_idx - 2)
                            prev_frame_idx = ((frame_idx - 2) // r) * r # taking stride over previous frames
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                        else:
                            # first find the nearest frame among every r-th frames after this frame
                            # for r=1, this would be (frame_idx + 2)
                            prev_frame_idx = -(-(frame_idx + 2) // r) * r
                            # then seek further among every r-th frames
                            prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                    out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if out is None:
                        # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                        # frames, we still attend to it as if it's a non-conditioning frame.
                        out = unselected_cond_outputs.get(prev_frame_idx, None)
                    t_pos_and_prevs.append((t_pos, out, False))

            for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].cuda(non_blocking=True)
                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_prompt_mask.append(
                    torch.zeros(B, seq_len, device=device, dtype=bool)
                )
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].cuda()
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):
                    # add a spatial embedding for the conditioning frame
                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding

                # Temporal positional encoding
                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                )
                to_cat_prompt_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            # Optionally, select only a subset of spatial memory frames during trainining
            if (
                self.training
                and self.prob_to_dropout_spatial_mem > 0
                and self.rng.random() < self.prob_to_dropout_spatial_mem
            ):
                num_spatial_mem_keep = self.rng.integers(len(to_cat_prompt) + 1)
                keep = self.rng.choice(
                    range(len(to_cat_prompt)), num_spatial_mem_keep, replace=False
                ).tolist()
                to_cat_prompt = [to_cat_prompt[i] for i in keep]
                to_cat_prompt_mask = [to_cat_prompt_mask[i] for i in keep]
                to_cat_prompt_pos_embed = [to_cat_prompt_pos_embed[i] for i in keep]

            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            # First add those object pointers from selected conditioning frames
            # (optionally, only include object pointers in the past during evaluation)
            if not self.training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_ptrs = [
                # Temporal pos encoding contains how far away each pointer is from current frame
                (
                    (frame_idx - t) * tpos_sign_mul,
                    out["obj_ptr"],
                    True,  # is_selected_cond_frame
                )
                for t, out in ptr_cond_outputs.items()
            ]

            # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if not self.use_memory_selection:
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                else:
                    if -t_diff <= -len(valid_indices):
                        break
                    t = valid_indices[-t_diff]

                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))

            # If we have at least one object pointer, add them to the across attention
            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)
                # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                obj_ptrs = torch.stack(ptrs_list, dim=0)
                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = (
                        obj_ptrs
                        + self.cond_frame_obj_ptr_embedding
                        * torch.tensor(is_selected_cond_frame_list, device=device)[
                            ..., None, None
                        ].float()
                    )
                # a temporal positional embedding based on how far each object pointer is from
                # the current frame (sine embedding normalized by the max pointer num).
                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                    device=device,
                )
                # expand to batch size
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:
                    # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)  # "to_cat_prompt_mask" is not used
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

            # Use a dummy token on the first grame (to avoid emtpy memory input to tranformer encoder)
            to_cat_prompt = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_prompt_mask = [torch.zeros(B, 1, device=device, dtype=bool)]
            to_cat_prompt_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None  # For now, we always masks are zeros anyways
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        output_dict=None,
        is_init_cond_frame=False,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        if is_mask_from_pts and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            pix_feat = pix_feat.view_as(pix_feat)
            maskmem_out = self.maskmem_backbone(
                pix_feat, mask_for_mem, skip_mask_sigmoid=True
            )
        else:
            maskmem_out = self.maskmem_backbone(image, pix_feat, mask_for_mem)
        # Clone the feats and pos_enc to enable compilation
        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        is_obj_appearing = (object_score_logits > 0).float()
        maskmem_features += (
            1 - is_obj_appearing[..., None, None]
        ) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        return maskmem_features, maskmem_pos_enc

    def forward_tracking(self, backbone_out, input, return_dict=False):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            img_ids = input.find_inputs[stage_id].img_ids
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_image = input.img_batch[img_ids]
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    current_image,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(input.img_batch, img_ids)
            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        image,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
        inference_state=None,
        freq_output_dict=None
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None:
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
                inference_state= inference_state,
                freq_output_dict=freq_output_dict
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, the SAM mask decoder should have `self.iter_use_prev_mask_pred=True`, and
            # any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert self.iter_use_prev_mask_pred
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                inference_state=inference_state,
                frame_idx=frame_idx
            )
        (
            _,_,_,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score,
            kf_ious,
            rvcot_ious
        ) = sam_outputs
        # Use the final prediction (after all correction steps for output and eval)        
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["best_iou_score"] = best_iou_score
        current_out["kf_ious"] = kf_ious
        current_out["rvcot_ious"] = rvcot_ious

        if self.use_memory_selection:
            current_out["object_score_logits"] = object_score_logits
            # iou_score = ious.max(-1)[0]
            iou_score = best_iou_score
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(
                object_score_logits, iou_score
            )
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        # (note that `self.num_maskmem == 0` is primarily used for reproducing SAM on
        # images, in which case we'll just skip memory encoder to save compute).
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        # Optionally, offload the outputs to CPU memory during evaluation to avoid
        # GPU OOM on very long videos or very large resolution or too many objects
        if self.offload_output_to_cpu_for_eval and not self.training:
            # Here we only keep those keys needed for evaluation to get a compact output
            trimmed_out = {
                "pred_masks": current_out["pred_masks"].cpu(),
                "pred_masks_high_res": current_out["pred_masks_high_res"].cpu(),
                # other items for evaluation (these are small tensors so we keep them on GPU)
                "obj_ptr": current_out["obj_ptr"],
                "object_score_logits": current_out["object_score_logits"],
            }
            if run_mem_encoder and self.num_maskmem > 0:
                trimmed_out["maskmem_features"] = maskmem_features.cpu()
                trimmed_out["maskmem_pos_enc"] = [x.cpu() for x in maskmem_pos_enc]
            if self.use_memory_selection:
                trimmed_out["iou_score"] = current_out["iou_score"].cpu()
                trimmed_out["eff_iou_score"] = current_out["eff_iou_score"].cpu()
            current_out = trimmed_out

        # Optionally, trim the output of past non-conditioning frame (r * num_maskmem frames
        # before the current frame) during evaluation. This is intended to save GPU or CPU
        # memory for semi-supervised VOS eval, where only the first frame receives prompts.
        def _trim_past_out(past_out, current_out):
            if past_out is None:
                return None
            return {
                "pred_masks": past_out["pred_masks"],
                "obj_ptr": past_out["obj_ptr"],
                "object_score_logits": past_out["object_score_logits"],
            }

        if self.trim_past_non_cond_mem_for_eval and not self.training:
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx, None)

            if past_out is not None:
                print(past_out.get("eff_iou_score", 0))
                if (
                    self.use_memory_selection
                    and past_out.get("eff_iou_score", 0) < self.mf_threshold
                ) or not self.use_memory_selection:
                    output_dict["non_cond_frame_outputs"][past_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

            if (
                self.use_memory_selection and not self.offload_output_to_cpu_for_eval
            ):  ## design for memory selection, trim too old frames to save memory
                far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
                past_out = output_dict["non_cond_frame_outputs"].get(
                    far_old_frame_idx, None
                )
                if past_out is not None:
                    output_dict["non_cond_frame_outputs"][far_old_frame_idx] = (
                        _trim_past_out(past_out, current_out)
                    )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _compile_all_components(self):
        """Compile all model components for faster inference."""
        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        from sam3.perflib.compile import compile_wrapper

        logging.info("Compiling all components. First time may be very slow.")

        self.maskmem_backbone.forward = compile_wrapper(
            self.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )
        self.transformer.encoder.forward = compile_wrapper(
            self.transformer.encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=True,  # Num. of memories varies
        )
        # We disable compilation of sam_prompt_encoder as it sometimes gives a large accuracy regression,
        # especially when sam_mask_prompt (previous mask logits) is not None
        # self.sam_prompt_encoder.forward = torch.compile(
        #     self.sam_prompt_encoder.forward,
        #     mode="max-autotune",
        #     fullgraph=True,
        #     dynamic=False,  # Accuracy regression on True
        # )
        self.sam_mask_decoder.forward = compile_wrapper(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

    def _maybe_clone(self, x):
        """Clone a tensor if and only if `self.compile_all_components` is True."""
        return x.clone() if self.compile_all_components else x


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}
