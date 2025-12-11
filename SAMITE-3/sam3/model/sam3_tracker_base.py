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


import torch.nn as nn
import sys
import numpy as np
from sam3.model.utils.kalman_filter import KalmanFilter
from einops import rearrange



def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels, dtype_override)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


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
        # Whether to use SAMURAI
        samurai_mode: bool = False,
        # Hyperparameters for SAMURAI
        stable_frames_threshold: int = 15,
        stable_ious_threshold: float = 0.3,
        min_obj_score_logits: float = -1,
        kf_score_weight: float = 0.15,
        memory_bank_iou_threshold: float = 0.5,
        memory_bank_obj_score_threshold: float = 0.0,
        memory_bank_kf_score_threshold: float = 0.0,
        # Whether to use SAMITE
        samite_mode : bool = True,
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

        # Whether to use original SAM 2, SAMURAI or SAMITE
        self.samurai_mode = samurai_mode
        self.samite_mode = samite_mode

        # Init Kalman Filter
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0
        self.use_second_prior_iou_thr = 0.8
        
        # Debug purpose
        self.history = {} # debug
        self.frame_cnt = 0 # debug

        # Hyperparameters for SAMURAI
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold
        
        # ========================================
        # Positional Encoding
        # ========================================
        self.pe = PositionalEncodingPermute2D(channels=256)

        if not (self.samurai_mode or self.samite_mode):
            print(f"\033[93mSAM 2 mode.\033[0m")
        elif self.samurai_mode:
            print(f"\033[93mSAMURAI mode.\033[0m")
        elif self.samite_mode:
            print(f"\033[SAMITE mode.\033[0m")


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


    def check_results(self, curr_feat, curr_mask, prev_feat, prev_mask, args):
        B = curr_feat.size(0)
        
        # current feats and preds
        curr_feat = curr_feat           # 3, c, h, w
        curr_mask = F.interpolate(curr_mask, size=prev_feat.size()[-2:], mode='bilinear')
        curr_mask = F.sigmoid(curr_mask)
        curr_mask[curr_mask >= 0.5] = 1
        curr_mask[curr_mask < 0.5] = 0  # 3, 1, h, w
        
        # previous feats and preds
        prev_feat = prev_feat.squeeze(0)                # n, c, h, w
        prev_mask = prev_mask.squeeze(0).unsqueeze(1)   # n, 1, h, w
        prev_num = prev_feat.size(0)
        
        # current pros
        curr_pro_fg = self.transformer.encoder.Weighted_GAP(curr_feat, curr_mask)       # 3, c, 1
        curr_pro_bg = self.transformer.encoder.Weighted_GAP(curr_feat, 1. - curr_mask)  # 3, c, 1
        
        curr_pro_fg = rearrange(curr_pro_fg, 'b c 1 n -> 1 c b n')  # 1, c, 3, 1
        curr_pro_bg = rearrange(curr_pro_bg, 'b c 1 n -> 1 c b n')  # 1, c, 3, 1
        
        curr_pro_fg = curr_pro_fg.expand(prev_num, -1, -1, -1)      # n, c, 3, 1
        curr_pro_bg = curr_pro_bg.expand(prev_num, -1, -1, -1)      # n, c, 3, 1
    
        # positional prior
        enc = self.pe(curr_feat)    # 3, c, h, w
        enc_priors = self.transformer.encoder.generate_prior_enc(enc, enc, [curr_mask], fts_size=prev_feat.size()[-2:])
        enc_priors = enc_priors[0]  # 3, 1, h, w
        enc_priors = rearrange(enc_priors, 'n 1 h w -> 1 n h w')  # 1, 3, h, w
        enc_priors = enc_priors.expand(prev_num, -1, -1, -1)      # n, 3, h, w
        
        # previous priors
        _, _, _, prev_prior = self.transformer.encoder.generate_prior_feat_enc_batch(prev_feat, curr_pro_fg, curr_pro_bg, enc_priors, fts_size=prev_feat.size()[-2:])
        thr = 0.5
        prev_prior[prev_prior >= thr] = 1
        prev_prior[prev_prior < thr] = 0
        
        # measure iou score between prior and gt
        prev_prior = prev_prior.to(torch.float32)  # n, 3, h, w
        prev_mask = prev_mask.to(torch.float32)    # n, 1, h, w
        
        best_ind = 0
        best_iou = -1
        ious = []
        for idx in range(B):
            intersection, union, _ = self.calculate_IoU(prev_prior[:, idx, ...].contiguous(), prev_mask[:, 0, ...].contiguous(), K=2)
            iou = intersection[1].cpu().numpy() / (union[1].cpu().numpy() + 1e-7)
            ious.append(iou)
            if iou >= best_iou:
                best_iou = iou
                best_ind = idx

        if best_iou < 0.7:  # beta
            best_ind = 0
                
        return best_iou, best_ind

        
    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        gt_masks=None,
        empty_first=False,  # when batch size is more than 1, whether embed first mask or not
        prev_feat=None,
        prev_mask=None,
        args=None,
        prior=None,
        update_kalman=True
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
            empty_first=empty_first  # when batch size is more than 1, whether embed first mask or not
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
        batch_inds = torch.arange(B, device=device)
        if multimask_output and (self.samurai_mode or self.samite_mode):  # SAMURAI
            if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
                best_iou_inds = torch.argmax(ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)  # b, 1, h, w
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
                # ========================================
                # select best mask along batch dimension
                # ========================================
                if empty_first and prev_feat is not None and prev_mask is not None:
                    if B in [2, 3]:
                        best_iou, best_ind = self.check_results(
                            curr_feat=backbone_features,
                            curr_mask=high_res_masks,
                            prev_feat=prev_feat,
                            prev_mask=prev_mask,
                            args=args,
                        )

                        ious = ious[best_ind: best_ind + 1, ...]
                        low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                        high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                        low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                        high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                        sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                        sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                        object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                        best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                        # if self.pred_obj_scores:
                        is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                    else:
                        print("Currently only support bs 2 or 3.")
                        sys.exit(0)
                
                if update_kalman:
                    non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)  # use sample 0 to update kalman statistics
                    if len(non_zero_indices) == 0:
                        high_res_bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))
                    self.frame_cnt += 1
                    self.stable_frames += 1
            elif self.stable_frames < self.stable_frames_threshold:
                if update_kalman:
                    self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                best_iou_inds = torch.argmax(ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                   
                # ========================================
                # select best mask along batch dimension
                # ========================================
                if empty_first and prev_feat is not None and prev_mask is not None:
                    if B in [2, 3]:
                        best_iou, best_ind = self.check_results(
                            curr_feat=backbone_features,
                            curr_mask=low_res_masks,
                            prev_feat=prev_feat,
                            prev_mask=prev_mask,
                            args=args,
                        )
                        
                        ious = ious[best_ind: best_ind + 1, ...]
                        low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                        high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                        low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                        high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                        sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                        sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                        object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                        best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                        # if self.pred_obj_scores:
                        is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                    else:
                        print("Currently only support bs 2 or 3.")
                        sys.exit(0)
                
                if update_kalman:
                    non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                    if len(non_zero_indices) == 0:
                        high_res_bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(dim=0).values
                        y_max, x_max = non_zero_indices.max(dim=0).values
                        high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                    if ious[0][best_iou_inds[0]] > self.stable_ious_threshold:
                        self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                        self.stable_frames += 1
                    else:
                        self.stable_frames = 0
                    self.frame_cnt += 1
            else:
                if update_kalman:
                    self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
                high_res_multibboxes = []
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
                # weighted iou
                weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
                if B > 1:  # do not have kalman filter for frame 1
                    weighted_ious[1:, ...] = ious[1:, ...]
                best_iou_inds = torch.argmax(weighted_ious, dim=-1)
                low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                if sam_output_tokens.size(1) > 1:
                    sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
                # ========================================
                # select best mask along batch dimension
                # ========================================
                if empty_first and prev_feat is not None and prev_mask is not None:
                    if B in [2, 3]:
                        best_iou, best_ind = self.check_results(
                            curr_feat=backbone_features,
                            curr_mask=low_res_masks,
                            prev_feat=prev_feat,
                            prev_mask=prev_mask,
                            args=args,
                        )
                        
                        ious = ious[best_ind: best_ind + 1, ...]
                        low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                        high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                        low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                        high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                        sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                        sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                        object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                        best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                        # if self.pred_obj_scores:
                        is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                    else:
                        print("Currently only support bs 2 or 3.")
                        sys.exit(0)
                
                if update_kalman:
                    self.frame_cnt += 1

                    if ious[0][best_iou_inds[0]] < self.stable_ious_threshold:
                        self.stable_frames = 0
                    else:
                        self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds[0]]))
            
        elif multimask_output and not (self.samurai_mode or self.samite_mode):  # original SAM 2 / SAMITE
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                
            # ========================================
            # select best mask along batch dimension
            # ========================================
            if empty_first and prev_feat is not None and prev_mask is not None:
                if B in [2, 3]:
                    best_iou, best_ind = self.check_results(
                        curr_feat=backbone_features,
                        curr_mask=low_res_masks,
                        prev_feat=prev_feat,
                        prev_mask=prev_mask,
                        args=args,
                    )
                    
                    ious = ious[best_ind: best_ind + 1, ...]
                    low_res_multimasks = low_res_multimasks[best_ind: best_ind + 1, ...]
                    high_res_multimasks = high_res_multimasks[best_ind: best_ind + 1, ...]
                    low_res_masks = low_res_masks[best_ind: best_ind + 1, ...]
                    high_res_masks = high_res_masks[best_ind: best_ind + 1, ...]
                    sam_output_token = sam_output_token[best_ind: best_ind + 1, ...]
                    sam_output_tokens = sam_output_tokens[best_ind: best_ind + 1, ...]
                    object_score_logits = object_score_logits[best_ind: best_ind + 1, ...]
                    best_iou_inds = best_iou_inds[best_ind: best_ind + 1, ...]
                    # if self.pred_obj_scores:
                    is_obj_appearing = is_obj_appearing[best_ind: best_ind + 1, ...]
                else:
                    print("Currently only support bs 2 or 3.")
                    sys.exit(0)    
        else:
            best_iou_inds = torch.tensor(0).to(device).view(1)
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks
            if (self.samurai_mode or self.samite_mode):
                non_zero_indices = torch.argwhere(high_res_masks[0][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                    high_res_bbox = [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
                self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))


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
            ious[0, best_iou_inds[0]],
            kf_ious[best_iou_inds[0]] if kf_ious is not None else None,
            prior[0] if prior is not None else None
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
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        # produce an object pointer using the SAM decoder from the mask input
        _, _, _, _, _, obj_ptr, _, _, _, _ = self._forward_sam_heads(
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
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
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
        args=None
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

        pro_fgs, pro_bgs, masks = [], [], []
        prior = None
        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame and use_prev_mem_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_prompt, to_cat_prompt_mask, to_cat_prompt_pos_embed = [], [], []
            to_cat_prev_feat = []
            to_cat_pro_fg, to_cat_pro_bg = [], []
            to_cat_memory_mask = []
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
            valid_indices_complete = []
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = 1 if self.training else self.memory_temporal_stride_for_eval

            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )
            
            if self.samurai_mode or self.samite_mode:
                valid_indices = []
                feats = []
                max_num_indices = args.max_num_indices if args is not None else 30
                if frame_idx > 1:
                    # ========================================
                    # Select valid previous frames as candidates
                    # - iou score is valid;
                    # - object appearing score is valid;
                    # - kalman motion score is valid.
                    # ========================================
                    for i in range(frame_idx - 1, 1, -1):  # Iterate backwards through previous frames
                        iou_score = output_dict["non_cond_frame_outputs"][i]["best_iou_score"]  # Get mask affinity score
                        obj_score = output_dict["non_cond_frame_outputs"][i]["object_score_logits"]  # Get object score
                        kf_score = output_dict["non_cond_frame_outputs"][i]["kf_score"] if "kf_score" in output_dict["non_cond_frame_outputs"][i] else None  # Get motion score if available
                        # Check if the scores meet the criteria for being a valid index
                        if iou_score.mean() > self.memory_bank_iou_threshold and \
                        obj_score.mean() > self.memory_bank_obj_score_threshold and \
                        (kf_score is None or kf_score.item() > self.memory_bank_kf_score_threshold):
                            # indices
                            valid_indices_complete.insert(0, i)  # all valid indices that satisfies kalman threshold
                            valid_indices.insert(0, i)  # compared to inds, valid_indices will be filtered
                            
                            # features
                            feats.insert(0, output_dict["non_cond_frame_outputs"][i]["image_features"].to(device, non_blocking=True).unsqueeze(1))  # b, 1, c, h, w
                            
                            # prototypes
                            pro_fgs.insert(0, output_dict["non_cond_frame_outputs"][i]["pro_fg"])
                            pro_bgs.insert(0, output_dict["non_cond_frame_outputs"][i]["pro_bg"])
                            
                            # masks
                            mask = output_dict["non_cond_frame_outputs"][i]["pred_masks"].to(device, non_blocking=True)
                            mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)  # (b=1, c=1, h=32, w=32)
                            mask = F.sigmoid(mask)
                            mask[mask >= 0.5] = 1
                            mask[mask < 0.5] = 0
                            masks.insert(0, mask)
                            
                        if len(valid_indices) >= max_num_indices:
                            break
                    
                    args.selection_strategy = 's2l_pos_feat_v2'
                    # ========================================
                    # Select good candidates
                    # ========================================
                    num_maskmem = self.num_maskmem - len(t_pos_and_prevs)  # already has 1 cond frame (frame #0, with gt bbox as input)
                    if len(pro_fgs) > num_maskmem:
                        pro_fg = torch.cat(pro_fgs, dim=2).squeeze(-1)  # 1, 256, n
                        cos_eps = 1e-7

                        if args.selection_strategy == "s2l_pos_feat_v2":
                            ind_last = valid_indices[-1]
                            q = pro_fg[:, :, :-1].transpose(-2, -1)             # 1, n-1, 256
                            k1 = pro_fg[:, :, -1:]                              # 1, 256, 1, last frame (pos)
                            k2 = t_pos_and_prevs[0][1]["pro_fg"].squeeze(-1)    # 1, 256, 1, first frame (feat)
                            
                            q_norm = torch.norm(q, 2, 2, True)      # 1, n-1, 256
                            k1_norm = torch.norm(k1, 2, 1, True)    # 1, 256, 1
                            k2_norm = torch.norm(k2, 2, 1, True)    # 1, 256, 1
                            
                            cos1 = (q @ k1) / (q_norm @ k1_norm + cos_eps)  # 1, n-1, 1
                            cos2 = (q @ k2) / (q_norm @ k2_norm + cos_eps)  # 1, n-1, 1
                            cos1 = cos1.squeeze(-1)  # 1, n-1
                            cos2 = cos2.squeeze(-1)  # 1, n-1
                            
                            alpha = args.alpha
                            cos = cos1 * alpha + cos2 * (1. - alpha)
                            
                            vals, inds = cos.topk(num_maskmem - 1, dim=1, largest=True, sorted=True)  # 1, num_maskmem - 1
                            inds = inds.squeeze(0)  # num_maskmem - 1
                            inds = inds.sort(descending=False, dim=0)[0]  # sort indices
                            
                            valid_indices = torch.tensor(valid_indices).to(q.device)
                            valid_indices = valid_indices[inds]  # select most similar frames
                            
                            valid_indices = valid_indices.detach().cpu().numpy().tolist()
                            valid_indices.append(ind_last)
                        else:
                            print("The selection strategy should be s2l_pos_feat_v2.")
                            sys.exit(0)
                    
                    self.valid_indices = valid_indices
                    
                args.frame_idx = frame_idx
                args.valid_indices = valid_indices

                # ========================================
                # Prepare to process selected candidates
                # ========================================
                start_idx = len(t_pos_and_prevs)
                end_idx = self.num_maskmem
                for t_pos in range(start_idx, end_idx):  # Iterate over the number of mask memories
                    idx = t_pos - self.num_maskmem  # Calculate the index for valid indices
                    if idx < -len(valid_indices):  # Skip if index is out of bounds
                        continue
                    out = output_dict["non_cond_frame_outputs"].get(valid_indices[idx], None)  # Get output for the valid index
                    if out is None:  # If not found, check unselected outputs
                        out = unselected_cond_outputs.get(valid_indices[idx], None)
                    t_pos_and_prevs.append((t_pos, out, False))  # Append the temporal position and output to the list
            else:
                for t_pos in range(1, self.num_maskmem):
                    t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                    if self.use_memory_selection:
                        if t_rel > len(valid_indices):
                            continue
                        prev_frame_idx = valid_indices[-t_rel]
                    else:
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
                                prev_frame_idx = ((frame_idx - 2) // r) * r
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


            # ========================================
            # Fetch useful information of selected candidates
            # ========================================
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

                # memory mask {0, 1}
                mask = prev["pred_masks"].to(device, non_blocking=True)
                mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)  # (b=1, c=1, h=32, w=32)
                mask = F.sigmoid(mask)
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
                to_cat_memory_mask.append(mask)
                
                # features
                to_cat_prev_feat.append(prev["image_features"].to(device, non_blocking=True).unsqueeze(1))
                
                # prototypes
                pro_fg = prev["pro_fg"].to(device, non_blocking=True)
                pro_bg = prev["pro_bg"].to(device, non_blocking=True)
                to_cat_pro_fg.append(pro_fg)
                to_cat_pro_bg.append(pro_bg)

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
            to_cat_memory_mask = None
            to_cat_prev_feat = None
            to_cat_pro_fg = None
            to_cat_pro_bg = None

        # Step 2: Concatenate the memories and forward through the transformer encoder
        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_mask = None  # For now, we always masks are zeros anyways
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        
        memory_mask = torch.cat(to_cat_memory_mask, dim=1) if to_cat_memory_mask is not None else None
        prev_feat = torch.cat(to_cat_prev_feat, dim=1) if to_cat_prev_feat is not None else None  # b, n, c, h, w

        # ========================================
        # Prior-Guided Memory Attention
        # ========================================
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,            # fg memory
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
            
            memory_mask=memory_mask,  # memory mask with a shape of (b, n, h, w) or None
            pro_fg=to_cat_pro_fg,     # fg prototypes
            pro_bg=to_cat_pro_bg,     # bg prototypes
            args=args,
            return_prior=True
        )
        prior = encoder_out['priors_']
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)

        if self.samite_mode and frame_idx > 0:
            if prior is not None:
                if not isinstance(prior, list):
                    prior = F.interpolate(prior, size=(self.image_size, self.image_size), mode='bilinear')
                    prior = prior * 20. - 10.
                else:
                    prior = [F.interpolate(p, size=(self.image_size, self.image_size), mode='bilinear') for p in prior]
                    prior = [p * 20. - 10. for p in prior]
                    prior = [prior[1]]
                    
            return pix_feat_with_mem, prior, prev_feat, memory_mask 
        else:
            return pix_feat_with_mem


    def calculate_IoU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
        area_output = torch.histc(output, bins=K, min=0, max=K - 1)
        area_target = torch.histc(target, bins=K, min=0, max=K - 1)
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target


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
        args=None
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
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
                args=args
            )

            
            if self.samite_mode and frame_idx > 0:
                pix_feat, prior, prev_feat, prev_mask = pix_feat
            else:
                prior = None
                prev_feat = None
                prev_mask = None
            
            
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, the SAM mask decoder should have `self.iter_use_prev_mask_pred=True`, and
            # any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert self.iter_use_prev_mask_pred
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            
            if args is not None and args.use_prior_prompt:
                mask_inputs = prior
            
            # ========================================
            # Batch
            # ========================================
            empty_first = False
            if prior is not None and isinstance(prior, list):
                empty_first = True
                if len(prior) == 1 or len(prior) == 2:
                    # copy pixel feat by 2 times 
                    pix_feat = pix_feat.expand(1 + len(prior), -1, -1, -1).contiguous()  # b=2/3, c, h, w
                    
                    # 3 prior masks
                    prior_0 = [torch.zeros_like(prior[-1])]
                    mask_inputs = torch.cat(prior_0 + prior, dim=0)  # b=2/3, 1, h, w
                
                    # copy each high res feature by 3 times
                    if high_res_features is not None:
                        high_res_features = [feat.expand(1 + len(prior), -1, -1, -1) for feat in high_res_features]
                else:
                    print("Currently only support bs 2.")
                    sys.exit(0)
            elif prior is not None and not isinstance(prior, list):  # post calibration
                print("Current only support prior list.")
                sys.exit(0)
            

            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)

            with torch.no_grad():
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    empty_first=empty_first,
                    prev_feat=prev_feat,
                    prev_mask=prev_mask,
                    args=args,
                    prior=prior,
                    update_kalman=True
                )
        (
            _,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score,
            kf_ious,
            prior
        ) = sam_outputs
        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["best_iou_score"] = best_iou_score
        current_out["kf_ious"] = kf_ious
        if self.use_memory_selection:
            current_out["object_score_logits"] = object_score_logits
            iou_score = ious.max(-1)[0]
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
            # fg memory
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
            # fg memory
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
