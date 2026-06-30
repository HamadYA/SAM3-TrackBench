# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

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
    ):
        """
        Forward SAM prompt encoders and mask heads.

        SAM2Long addition: return `obj_ptrs`, the object pointers for all SAM
        output tokens/candidates, in addition to the best/single `obj_ptr`.
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
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
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
                repeat_image=False,
                high_res_features=high_res_features,
            )

        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)
        object_score_logits = self._maybe_clone(object_score_logits)

        if self.training and getattr(self, "teacher_force_obj_scores_for_mem", False):
            assert gt_masks is not None
            is_obj_appearing = torch.any(gt_masks.float().flatten(1) > 0, dim=1)
            is_obj_appearing = is_obj_appearing[..., None]
        else:
            is_obj_appearing = object_score_logits > 0

        if is_obj_appearing.ndim == 1:
            is_obj_appearing = is_obj_appearing[:, None]

        # Mask used for spatial memories is always a hard obj/no-obj choice.
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Best/single object pointer.
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        # SAM2Long: object pointers for all output tokens/candidates.
        obj_ptrs = self.obj_ptr_proj(sam_output_tokens)
        lambda_for_obj_ptrs = lambda_is_obj_appearing
        while lambda_for_obj_ptrs.ndim < obj_ptrs.ndim:
            lambda_for_obj_ptrs = lambda_for_obj_ptrs.unsqueeze(1)
        obj_ptrs = lambda_for_obj_ptrs * obj_ptrs
        obj_ptrs = obj_ptrs + (1 - lambda_for_obj_ptrs) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            obj_ptrs,
        )


    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into output mask logits without using SAM.

        SAM2Long compatibility: returns an 8th value, `obj_ptrs`, which is None
        for mask-input frames.
        """
        out_scale, out_bias = 20.0, -10.0
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
            antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()

        _, _, _, _, _, obj_ptr, _, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )

        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
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
        mem_pick_index=0,
        start_frame_idx=0,
        iou_thre=0.1,
    ):
        """Fuse the current frame's visual feature map with previous memory.

        This keeps SAM3's encoder/prompt structure and copies SAM2Long's long
        memory selection, candidate-indexed memory retrieval, and score-modulated
        memory/object-pointer bookkeeping.
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        def _sam2long_enabled(index):
            if index is None:
                return False
            if isinstance(index, int):
                return index != 0
            if torch.is_tensor(index) and index.ndim == 0:
                return int(index.item()) != 0
            return True

        def _get_mem_pick(index, frame_id):
            if not _sam2long_enabled(index):
                return 0
            if isinstance(index, dict):
                return int(index.get(frame_id, 0))
            try:
                value = index[frame_id]
            except Exception:
                return 0
            if torch.is_tensor(value):
                value = value.item()
            return int(value)

        use_sam2long = _sam2long_enabled(mem_pick_index)
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        object_frame_score = None
        object_ptr_score = None

        if not is_init_cond_frame and use_prev_mem_frame:
            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []
            assert len(output_dict["cond_frame_outputs"]) > 0
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx,
                cond_outputs,
                self.max_cond_frames_in_attn,
                keep_first_cond_frame=self.keep_first_cond_frame,
            )

            ref_frame_idx = start_frame_idx if start_frame_idx in cond_outputs else next(iter(cond_outputs))
            num_object = int(cond_outputs[ref_frame_idx]["obj_ptr"].shape[0])
            score_device = cond_outputs[ref_frame_idx]["obj_ptr"].device
            score_dtype = cond_outputs[ref_frame_idx]["obj_ptr"].dtype

            # SAM2Long uses t_pos=0 for conditioning spatial memories.
            t_pos_and_prevs = [
                (0, out, True, t) for t, out in selected_cond_outputs.items()
            ]

            r = 1 if self.training else self.memory_temporal_stride_for_eval
            max_obj_ptrs_in_encoder = (
                min(num_frames, self.max_obj_ptrs_in_encoder)
                if num_frames is not None
                else self.max_obj_ptrs_in_encoder
            )

            if use_sam2long:
                valid_indices = []
                if not track_in_reverse:
                    if frame_idx > start_frame_idx + 1:
                        for i in range(frame_idx - 1, start_frame_idx, -1):
                            prev_out = output_dict["non_cond_frame_outputs"].get(i, None)
                            if prev_out is None:
                                continue
                            mem_idx = _get_mem_pick(mem_pick_index, i)
                            object_score = prev_out["object_score_logits"][..., mem_idx]
                            iou = prev_out["ious"][..., mem_idx]
                            if iou.float().mean().item() > iou_thre and object_score.float().mean().item() > 0:
                                valid_indices.insert(0, i)
                            if len(valid_indices) >= max_obj_ptrs_in_encoder - 1:
                                break
                        if frame_idx - 1 not in valid_indices:
                            valid_indices.append(frame_idx - 1)
                else:
                    # Reverse-time analogue of the SAM2Long scan.
                    if frame_idx < start_frame_idx - 1:
                        for i in range(frame_idx + 1, start_frame_idx):
                            prev_out = output_dict["non_cond_frame_outputs"].get(i, None)
                            if prev_out is None:
                                continue
                            mem_idx = _get_mem_pick(mem_pick_index, i)
                            object_score = prev_out["object_score_logits"][..., mem_idx]
                            iou = prev_out["ious"][..., mem_idx]
                            if iou.float().mean().item() > iou_thre and object_score.float().mean().item() > 0:
                                valid_indices.insert(0, i)
                            if len(valid_indices) >= max_obj_ptrs_in_encoder - 1:
                                break
                        if frame_idx + 1 not in valid_indices:
                            valid_indices.append(frame_idx + 1)
            elif self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )
            else:
                valid_indices = None

            # Add non-conditioning memory frames.
            for t_pos in range(1, self.num_maskmem):
                if use_sam2long:
                    idx = t_pos - self.num_maskmem
                    if valid_indices is None or idx < -len(valid_indices):
                        continue
                    prev_frame_idx = valid_indices[idx]
                else:
                    t_rel = self.num_maskmem - t_pos
                    if self.use_memory_selection:
                        if valid_indices is None or t_rel > len(valid_indices):
                            continue
                        prev_frame_idx = valid_indices[-t_rel]
                    else:
                        if t_rel == 1:
                            prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
                        else:
                            if not track_in_reverse:
                                prev_frame_idx = ((frame_idx - 2) // r) * r
                                prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                            else:
                                prev_frame_idx = -(-(frame_idx + 2) // r) * r
                                prev_frame_idx = prev_frame_idx + (t_rel - 2) * r

                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out, False, prev_frame_idx))

            if use_sam2long:
                object_frame_score = []

            # Spatial memories.
            for t_pos, prev, is_selected_cond_frame, prev_idx in t_pos_and_prevs:
                if prev is None:
                    continue

                if use_sam2long and t_pos > 0:
                    mem_idx = _get_mem_pick(mem_pick_index, prev_idx)
                    object_frame_score.append(
                        prev["object_score_logits"][..., mem_idx].view(-1)
                    )
                    feats = prev["maskmem_features"][..., mem_idx].to(
                        device=device, non_blocking=True
                    )
                else:
                    if use_sam2long:
                        object_frame_score.append(
                            torch.ones(num_object, device=score_device, dtype=score_dtype).mul_(10)
                        )
                    feats = prev["maskmem_features"].to(device=device, non_blocking=True)

                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_prompt_mask.append(
                    torch.zeros(B, seq_len, device=device, dtype=bool)
                )

                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device=device, non_blocking=True)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):
                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding

                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                to_cat_prompt_pos_embed.append(maskmem_enc)

            if (
                self.training
                and getattr(self, "prob_to_dropout_spatial_mem", 0) > 0
                and getattr(self, "rng", None) is not None
                and self.rng.random() < self.prob_to_dropout_spatial_mem
            ):
                num_spatial_mem_keep = self.rng.integers(len(to_cat_prompt) + 1)
                keep = self.rng.choice(
                    range(len(to_cat_prompt)), num_spatial_mem_keep, replace=False
                ).tolist()
                to_cat_prompt = [to_cat_prompt[i] for i in keep]
                to_cat_prompt_mask = [to_cat_prompt_mask[i] for i in keep]
                to_cat_prompt_pos_embed = [to_cat_prompt_pos_embed[i] for i in keep]
                if object_frame_score is not None:
                    object_frame_score = [object_frame_score[i] for i in keep]

            # Object pointers.
            if not self.training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs

            pos_and_ptrs = []
            if use_sam2long:
                object_ptr_score = []

            for t, out in ptr_cond_outputs.items():
                ptr = out.get("obj_ptr", None)
                if ptr is None:
                    continue
                pos = abs(frame_idx - t) if use_sam2long else (frame_idx - t) * tpos_sign_mul
                pos_and_ptrs.append((pos, ptr, True))
                if use_sam2long:
                    object_ptr_score.append(
                        torch.ones(num_object, device=score_device, dtype=score_dtype).mul_(10)
                    )

            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if use_sam2long or self.use_memory_selection:
                    if valid_indices is None or -t_diff <= -len(valid_indices):
                        break
                    t = valid_indices[-t_diff]
                else:
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break

                out = output_dict["non_cond_frame_outputs"].get(
                    t, unselected_cond_outputs.get(t, None)
                )
                if out is None or out.get("obj_ptr", None) is None:
                    continue

                if use_sam2long:
                    mem_idx = _get_mem_pick(mem_pick_index, t)
                    object_ptr_score.append(
                        out["object_score_logits"][..., mem_idx].view(-1)
                    )
                    ptr = out["obj_ptr"][..., mem_idx]
                else:
                    ptr = out["obj_ptr"]
                pos_and_ptrs.append((t_diff, ptr, False))

            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)
                obj_ptrs = torch.stack(ptrs_list, dim=0)

                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = (
                        obj_ptrs
                        + self.cond_frame_obj_ptr_embedding
                        * torch.tensor(is_selected_cond_frame_list, device=device)[
                            ..., None, None
                        ].float()
                    )

                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                    device=device,
                )
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)

                if self.mem_dim < C:
                    obj_ptrs = obj_ptrs.reshape(-1, B, C // self.mem_dim, self.mem_dim)
                    obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                    obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None
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
            object_frame_scores=object_frame_score,
            object_ptr_scores=object_ptr_score,
        )
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
        """Encode the current image and its prediction into a memory feature.

        SAM2Long compatibility: when `pred_masks_high_res` contains multiple mask
        candidates [B, M, H, W], encode every candidate and store the resulting
        memory as [B, C, Hmem, Wmem, M], so later memory conditioning can select
        `[..., mem_pick_index[frame]]`.
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

        candidate_mode = (
            pred_masks_high_res.ndim == 4
            and pred_masks_high_res.size(0) == B
            and pred_masks_high_res.size(1) > 1
        )
        num_candidates = pred_masks_high_res.size(1) if candidate_mode else 1

        if self.non_overlap_masks_for_mem_enc and not self.training:
            if candidate_mode:
                pred_masks_high_res = torch.cat(
                    [
                        self._apply_non_overlapping_constraints(
                            pred_masks_high_res[:, cand_idx : cand_idx + 1]
                        )
                        for cand_idx in range(num_candidates)
                    ],
                    dim=1,
                )
            else:
                pred_masks_high_res = self._apply_non_overlapping_constraints(
                    pred_masks_high_res
                )

        if is_mask_from_pts and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        object_score_logits_for_mem = object_score_logits
        if object_score_logits_for_mem.ndim == 1:
            object_score_logits_for_mem = object_score_logits_for_mem[:, None]

        if candidate_mode:
            mask_for_mem = mask_for_mem.reshape(
                B * num_candidates, 1, *mask_for_mem.shape[-2:]
            )
            pix_feat_for_mem = (
                pix_feat[:, None]
                .expand(B, num_candidates, C, H, W)
                .reshape(B * num_candidates, C, H, W)
            )
            if image is not None:
                image_for_mem = (
                    image[:, None]
                    .expand(B, num_candidates, *image.shape[1:])
                    .reshape(B * num_candidates, *image.shape[1:])
                )
            else:
                image_for_mem = None

            if (
                object_score_logits_for_mem.shape[0] == B
                and object_score_logits_for_mem.shape[-1] == num_candidates
            ):
                object_score_logits_for_mem = object_score_logits_for_mem.reshape(
                    B * num_candidates, 1
                )
            elif object_score_logits_for_mem.shape[0] == B:
                object_score_logits_for_mem = (
                    object_score_logits_for_mem[:, :1]
                    .expand(B, num_candidates)
                    .reshape(B * num_candidates, 1)
                )
            else:
                object_score_logits_for_mem = object_score_logits_for_mem.reshape(
                    B * num_candidates, -1
                )[:, :1]
        else:
            pix_feat_for_mem = pix_feat
            image_for_mem = image

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            maskmem_out = self.maskmem_backbone(
                pix_feat_for_mem,
                mask_for_mem,
                skip_mask_sigmoid=True,
            )
        else:
            maskmem_out = self.maskmem_backbone(
                image_for_mem,
                pix_feat_for_mem,
                mask_for_mem,
            )

        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]

        is_obj_appearing = (object_score_logits_for_mem > 0).float()
        maskmem_features += (
            1 - is_obj_appearing[..., None, None]
        ) * self.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        if candidate_mode:
            _, mem_c, mem_h, mem_w = maskmem_features.shape
            maskmem_features = (
                maskmem_features.view(B, num_candidates, mem_c, mem_h, mem_w)
                .permute(0, 2, 3, 4, 1)
                .contiguous()
            )
            new_maskmem_pos_enc = []
            for pos in maskmem_pos_enc:
                if pos.shape[0] == B * num_candidates:
                    pos = pos.view(B, num_candidates, *pos.shape[1:])[:, 0].contiguous()
                new_maskmem_pos_enc.append(pos)
            maskmem_pos_enc = new_maskmem_pos_enc

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
                # ==================================================================
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                # ==================================================================
                output_dict=output_dict,
                num_frames=num_frames,
                mem_pick_index=backbone_out.get("mem_pick_index", 0),
                start_frame_idx=backbone_out.get(
                    "start_frame_idx",
                    init_cond_frames[0] if len(init_cond_frames) > 0 else 0,
                ),
                iou_thre=backbone_out.get("iou_thre", 0.1),
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
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
        mem_pick_index=0,
        start_frame_idx=0,
        iou_thre=0.1,
        gt_masks=None,
        frames_to_add_correction_pt=None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        def _sam2long_enabled(index):
            if index is None:
                return False
            if isinstance(index, int):
                return index != 0
            if torch.is_tensor(index) and index.ndim == 0:
                return int(index.item()) != 0
            return True

        use_candidate_outputs = _sam2long_enabled(mem_pick_index)

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if mask_inputs is not None:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
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
                mem_pick_index=mem_pick_index,
                start_frame_idx=start_frame_idx,
                iou_thre=iou_thre,
            )
            if prev_sam_mask_logits is not None:
                assert self.iter_use_prev_mask_pred
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if use_candidate_outputs:
                multimask_output = True
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                gt_masks=gt_masks,
            )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            obj_ptrs,
        ) = sam_outputs

        num_candidates = ious.shape[-1] if ious.ndim > 1 else 1
        use_candidate_outputs = (
            use_candidate_outputs
            and obj_ptrs is not None
            and high_res_multimasks.ndim == 4
            and high_res_multimasks.shape[1] > 1
        )

        object_score_logits_for_output = object_score_logits
        if object_score_logits_for_output.ndim == 1:
            object_score_logits_for_output = object_score_logits_for_output[:, None]
        if (
            use_candidate_outputs
            and object_score_logits_for_output.shape[-1] == 1
            and num_candidates > 1
        ):
            object_score_logits_for_output = object_score_logits_for_output.expand(
                -1, num_candidates
            )

        if use_candidate_outputs:
            current_out["pred_masks"] = low_res_multimasks
            current_out["ious"] = ious
            current_out["object_score"] = object_score_logits_for_output[:, 0]
            obj_ptrs_for_output = obj_ptrs
            if obj_ptrs_for_output.ndim == 3:
                if obj_ptrs_for_output.shape[1] > num_candidates:
                    obj_ptrs_for_output = obj_ptrs_for_output[:, :num_candidates]
                if obj_ptrs_for_output.shape[1] == num_candidates:
                    obj_ptrs_for_output = obj_ptrs_for_output.permute(0, 2, 1).contiguous()
            current_out["obj_ptr"] = obj_ptrs_for_output
            current_out["pred_masks_high_res"] = high_res_multimasks
        else:
            current_out["pred_masks"] = low_res_masks
            current_out["ious"] = ious.max(-1)[0] if ious.ndim > 1 else ious
            current_out["object_score"] = object_score_logits_for_output[:, 0]
            current_out["obj_ptr"] = obj_ptr
            current_out["pred_masks_high_res"] = high_res_masks

        current_out["object_score_logits"] = object_score_logits_for_output

        if self.use_memory_selection:
            iou_score = ious.max(-1)[0] if ious.ndim > 1 else ious
            current_out["iou_score"] = iou_score
            iou_score_for_cal = iou_score
            if iou_score_for_cal.ndim == 1 and object_score_logits_for_output.ndim == 2:
                iou_score_for_cal = iou_score_for_cal[:, None]
            current_out["eff_iou_score"] = self.cal_mem_score(
                object_score_logits_for_output,
                iou_score_for_cal,
            )

        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = (
                high_res_multimasks if use_candidate_outputs else high_res_masks
            )
            object_score_logits_for_mem_enc = (
                object_score_logits_for_output
                if use_candidate_outputs
                else object_score_logits
            )
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits_for_mem_enc,
                is_mask_from_pts=(point_inputs is not None),
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if self.offload_output_to_cpu_for_eval and not self.training:
            trimmed_out = {
                "pred_masks": current_out["pred_masks"].cpu(),
                "pred_masks_high_res": current_out["pred_masks_high_res"].cpu(),
                "obj_ptr": current_out["obj_ptr"],
                "object_score_logits": current_out["object_score_logits"],
                "ious": current_out["ious"].cpu()
                if torch.is_tensor(current_out["ious"])
                else current_out["ious"],
                "object_score": current_out["object_score"].cpu()
                if torch.is_tensor(current_out["object_score"])
                else current_out["object_score"],
            }
            if run_mem_encoder and self.num_maskmem > 0:
                trimmed_out["maskmem_features"] = maskmem_features.cpu()
                trimmed_out["maskmem_pos_enc"] = [x.cpu() for x in maskmem_pos_enc]
            if self.use_memory_selection:
                trimmed_out["iou_score"] = current_out["iou_score"].cpu()
                trimmed_out["eff_iou_score"] = current_out["eff_iou_score"].cpu()
            current_out = trimmed_out

        def _trim_past_out(past_out, current_out):
            if past_out is None:
                return None
            kept = {
                "pred_masks": past_out["pred_masks"],
                "obj_ptr": past_out["obj_ptr"],
                "object_score_logits": past_out["object_score_logits"],
            }
            for key in ("ious", "object_score", "iou_score", "eff_iou_score", "acc_score"):
                if key in past_out:
                    kept[key] = past_out[key]
            return kept

        if self.trim_past_non_cond_mem_for_eval and not self.training:
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx, None)

            if past_out is not None:
                if (
                    self.use_memory_selection
                    and past_out.get("eff_iou_score", 0) < self.mf_threshold
                ) or not self.use_memory_selection:
                    output_dict["non_cond_frame_outputs"][past_frame_idx] = _trim_past_out(
                        past_out, current_out
                    )

            if self.use_memory_selection and not self.offload_output_to_cpu_for_eval:
                far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
                past_out = output_dict["non_cond_frame_outputs"].get(
                    far_old_frame_idx, None
                )
                if past_out is not None:
                    output_dict["non_cond_frame_outputs"][far_old_frame_idx] = _trim_past_out(
                        past_out, current_out
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
