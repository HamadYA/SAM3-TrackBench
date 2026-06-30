# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Transformer decoder.
Inspired from Pytorch's version, adds the pre-norm variant
"""

from typing import Any, Dict, List, Optional

import numpy as np

import torch

from sam3.sam.transformer import RoPEAttention

from torch import nn, Tensor
from torchvision.ops.roi_align import RoIAlign

from .act_ckpt_utils import activation_ckpt_wrapper

from .box_ops import box_cxcywh_to_xyxy

from .model_misc import (
    gen_sineembed_for_position,
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    MLP,
)

import torch.nn.functional as F
from einops import rearrange
import os
import sys
import cv2
import numpy as np


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        cross_attention: nn.Module,
        n_heads: int,
        use_text_cross_attention: bool = False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.amp.autocast(device_type="cuda", enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # num_token, bs, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        # dac
        dac=False,
        dac_use_selfatt_ln=True,
        presence_token=None,
        # skip inside deformable attn
        identity=0.0,
        **kwargs,  # additional kwargs for compatibility
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            if dac:
                # we only apply self attention to the first half of the queries
                assert tgt.shape[0] % 2 == 0
                num_o2o_queries = tgt.shape[0] // 2
                tgt_o2o = tgt[:num_o2o_queries]
                tgt_query_pos_o2o = tgt_query_pos[:num_o2o_queries]
                tgt_o2m = tgt[num_o2o_queries:]
            else:
                tgt_o2o = tgt
                tgt_query_pos_o2o = tgt_query_pos

            if presence_token is not None:
                tgt_o2o = torch.cat([presence_token, tgt_o2o], dim=0)
                tgt_query_pos_o2o = torch.cat(
                    [torch.zeros_like(presence_token), tgt_query_pos_o2o], dim=0
                )
                tgt_query_pos = torch.cat(
                    [torch.zeros_like(presence_token), tgt_query_pos], dim=0
                )

            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
            tgt2 = self.self_attn(q, k, tgt_o2o, attn_mask=self_attn_mask)[0]
            tgt_o2o = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    tgt_o2o = self.norm2(tgt_o2o)
                tgt = torch.cat((tgt_o2o, tgt_o2m), dim=0)  # Recombine
                if dac_use_selfatt_ln:
                    tgt = self.norm2(tgt)
            else:
                tgt = tgt_o2o
                tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if presence_token is not None:
            presence_token_mask = torch.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = torch.cat(
                [presence_token_mask, cross_attn_mask], dim=1
            )  # (bs*nheads, 1+nq, hw)

        # Cross attention to image
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1)
                if memory_key_padding_mask is not None
                else None
            ),
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:1]
            tgt = tgt[1:]

        return tgt, presence_token_out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        interaction_layer,
        layer,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        num_o2m_queries: int = 0,
        dac: bool = False,
        boxRPB: str = "none",
        # Experimental: An object query for SAM 2 tasks
        instance_query: bool = False,
        # Defines the number of additional instance queries,
        # 1 or 4 are the most likely for single vs multi mask support
        num_instances: int = 1,  # Irrelevant if instance_query is False
        dac_use_selfatt_ln: bool = True,
        use_act_checkpoint: bool = False,
        compile_mode=None,
        presence_token: bool = False,
        clamp_presence_logits: bool = True,
        clamp_presence_logit_max_val: float = 10.0,
        use_normed_output_consistently: bool = True,
        separate_box_head_instance: bool = False,
        separate_norm_instance: bool = False,
        resolution: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.fine_layers = (
            get_clones(interaction_layer, num_layers)
            if interaction_layer is not None
            else [None] * num_layers
        )
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        self.use_normed_output_consistently = use_normed_output_consistently

        self.instance_norm = nn.LayerNorm(d_model) if separate_norm_instance else None
        self.instance_bbox_embed = None
        if separate_box_head_instance:
            self.instance_bbox_embed = MLP(d_model, d_model, 4, 3)
        if instance_query:
            self.instance_query_embed = nn.Embedding(num_instances, d_model)
        self.box_refine = box_refine
        if box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

            self.reference_points = nn.Embedding(num_queries, 4)
            if instance_query:
                self.instance_reference_points = nn.Embedding(num_instances, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads

            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache = {}

            if resolution is not None and stride is not None:
                feat_size = resolution // stride
                coords_h, coords_w = self._get_coords(
                    feat_size, feat_size, device="cuda"
                )
                self.compilable_cord_cache = (coords_h, coords_w)
                self.compilable_stored_size = (feat_size, feat_size)

        self.roi_pooler = (
            RoIAlign(output_size=7, spatial_scale=1, sampling_ratio=-1, aligned=True)
            if interaction_layer is not None
            else None
        )
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.presence_token = None
        self.clamp_presence_logits = clamp_presence_logits
        self.clamp_presence_logit_max_val = clamp_presence_logit_max_val
        if presence_token:
            self.presence_token = nn.Embedding(1, d_model)
            self.presence_token_head = MLP(d_model, d_model, 1, 3)
            self.presence_token_out_norm = nn.LayerNorm(d_model)

        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln
        self.use_act_checkpoint = use_act_checkpoint

        nn.init.normal_(self.query_embed.weight.data)
        if self.instance_query_embed is not None:
            nn.init.normal_(self.instance_query_embed.weight.data)

        assert self.roi_pooler is None
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        self.compile_mode = compile_mode
        self.compiled = False
        # We defer compilation till after the first forward, to first warm-up the boxRPB cache

        # assign layer index to each layer so that some layers can decide what to do
        # based on which layer index they are (e.g. cross attention to memory bank only
        # in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W, device):
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
        bs, num_queries, _ = boxes_xyxy.shape
        if self.compilable_cord_cache is None:
            self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
            self.compilable_stored_size = (H, W)

        if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (
            H,
            W,
        ):
            # good, hitting the cache, will be compilable
            coords_h, coords_w = self.compilable_cord_cache
        else:
            # cache miss, will create compilation issue
            # In case we're not compiling, we'll still rely on the dict-based cache
            if feat_size not in self.coord_cache:
                self.coord_cache[feat_size] = self._get_coords(
                    H, W, reference_boxes.device
                )
            coords_h, coords_w = self.coord_cache[feat_size]

            assert coords_h.shape == (H,)
            assert coords_w.shape == (W,)

        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)

        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8  # normalize to -8, 8
            deltas_x_log = (
                torch.sign(deltas_x_log)
                * torch.log2(torch.abs(deltas_x_log) + 1.0)
                / np.log2(8)
            )

            deltas_y_log = deltas_y * 8  # normalize to -8, 8
            deltas_y_log = (
                torch.sign(deltas_y_log)
                * torch.log2(torch.abs(deltas_y_log) + 1.0)
                / np.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

        if self.training:
            assert self.use_act_checkpoint, "activation ckpt not enabled in decoder"
        deltas_x = activation_ckpt_wrapper(self.boxRPB_embed_x)(
            x=deltas_x,
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # bs, num_queries, W, n_heads
        deltas_y = activation_ckpt_wrapper(self.boxRPB_embed_y)(
            x=deltas_y,
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # bs, num_queries, H, n_heads

        if not torch.compiler.is_dynamo_compiling():
            assert deltas_x.shape[:3] == (bs, num_queries, W)
            assert deltas_y.shape[:3] == (bs, num_queries, H)

        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(
            2
        )  # bs, num_queries, H, W, n_heads
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[:4] == (bs, num_queries, H, W)
        B = B.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        B = B.permute(0, 3, 1, 2)  # bs, n_heads, num_queries, H*W
        B = B.contiguous()  # memeff attn likes ordered strides
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[2:] == (num_queries, H * W)
        return B

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        reference_boxes: Optional[Tensor] = None,  # num_queries, bs, 4
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        # if `apply_dac` is None, it will default to `self.dac`
        apply_dac: Optional[bool] = None,
        is_instance_prompt=False,
        decoder_extra_kwargs: Optional[Dict] = None,
        # ROI memory bank
        obj_roi_memory_feat=None,
        obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: \\sum{hw}, bs, d_model
            - pos: \\sum{hw}, bs, d_model
            - reference_boxes: nq, bs, 4 (after sigmoid)
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        if memory_mask is not None:
            assert (
                self.boxRPB == "none"
            ), "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"

        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert (tgt.shape[0] == self.num_queries) or (
                self.use_instance_query
                and (tgt.shape[0] == self.instance_query_embed.num_embeddings)
            )

            tgt = tgt.repeat(2, 1, 1)
            # note that we don't tile tgt_mask, since DAC doesn't
            # use self-attention in o2m queries
            if reference_boxes is not None:
                assert (reference_boxes.shape[0] == self.num_queries) or (
                    self.use_instance_query
                    and (
                        reference_boxes.shape[0]
                        == self.instance_query_embed.num_embeddings
                    )
                )
                reference_boxes = reference_boxes.repeat(2, 1, 1)

        bs = tgt.shape[1]
        intermediate = []
        intermediate_presence_logits = []
        presence_feats = None

        if self.box_refine:
            if reference_boxes is None:
                # In this case, we're in a one-stage model, so we generate the reference boxes
                reference_boxes = self.reference_points.weight.unsqueeze(1)
                reference_boxes = (
                    reference_boxes.repeat(2, bs, 1)
                    if apply_dac
                    else reference_boxes.repeat(1, bs, 1)
                )
                reference_boxes = reference_boxes.sigmoid()
            intermediate_ref_boxes = [reference_boxes]
        else:
            reference_boxes = None
            intermediate_ref_boxes = None

        output = tgt
        presence_out = None
        if self.presence_token is not None and is_instance_prompt is False:
            # expand to batch dim
            presence_out = self.presence_token.weight[None].expand(1, bs, -1)

        box_head = self.bbox_embed
        if is_instance_prompt and self.instance_bbox_embed is not None:
            box_head = self.instance_bbox_embed

        out_norm = self.norm
        if is_instance_prompt and self.instance_norm is not None:
            out_norm = self.instance_norm

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4

            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # nq, bs, d_model*2

            # conditional query
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, d_model

            if self.boxRPB != "none" and reference_boxes is not None:
                assert (
                    spatial_shapes.shape[0] == 1
                ), "only single scale support implemented"
                memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    (spatial_shapes[0, 0], spatial_shapes[0, 1]),
                )
                memory_mask = memory_mask.flatten(0, 1)  # (bs*n_heads, nq, H*W)
            if self.training:
                assert (
                    self.use_act_checkpoint
                ), "Activation checkpointing not enabled in the decoder"
            output, presence_out = activation_ckpt_wrapper(layer)(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                presence_token=presence_out,
                **(decoder_extra_kwargs or {}),
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                # ROI memory bank
                obj_roi_memory_feat=obj_roi_memory_feat,
                obj_roi_memory_mask=obj_roi_memory_mask,
            )

            # iter update
            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if box_head_trk is None:
                    # delta_unsig = self.bbox_embed(output)
                    if not self.use_normed_output_consistently:
                        delta_unsig = box_head(output)
                    else:
                        delta_unsig = box_head(out_norm(output))
                else:
                    # box_head_trk use a separate box head for tracking queries
                    Q_det = decoder_extra_kwargs["Q_det"]
                    assert output.size(0) >= Q_det
                    delta_unsig_det = self.bbox_embed(output[:Q_det])
                    delta_unsig_trk = box_head_trk(output[Q_det:])
                    delta_unsig = torch.cat([delta_unsig_det, delta_unsig_trk], dim=0)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_boxes = new_reference_points.detach()
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                raise NotImplementedError("not implemented yet")

            intermediate.append(out_norm(output))
            if self.presence_token is not None and is_instance_prompt is False:
                # norm, mlp head
                intermediate_layer_presence_logits = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)

                # clamp to mitigate numerical issues
                if self.clamp_presence_logits:
                    intermediate_layer_presence_logits.clamp(
                        min=-self.clamp_presence_logit_max_val,
                        max=self.clamp_presence_logit_max_val,
                    )

                intermediate_presence_logits.append(intermediate_layer_presence_logits)
                presence_feats = presence_out.clone()

        if not self.compiled and self.compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=self.compile_mode, fullgraph=True
            )
            self.compiled = True

        return (
            torch.stack(intermediate),
            torch.stack(intermediate_ref_boxes),
            (
                torch.stack(intermediate_presence_logits)
                if self.presence_token is not None and is_instance_prompt is False
                else None
            ),
            presence_feats,
        )


class TransformerEncoderCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,  # Do layers expect batch first input?
        # which layers to exclude cross attention? default: None, means all
        # layers use cross attention
        remove_cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint

        # ========================================

        # ========================================
        # Positional Encoding
        # ========================================
        self.pe = PositionalEncodingPermute2D(channels=256)
        
        # ========================================

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.batch_first = batch_first

        # remove cross attention layers if specified
        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)

        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None
                self.layers[i].dropout2 = None

    def forward(
        self,
        src,  # self-attention inputs
        prompt,  # cross-attention inputs
        src_mask: Optional[Tensor] = None,  # att.mask for self-attention inputs
        prompt_mask: Optional[Tensor] = None,  # att.mask for cross-attention inputs
        src_key_padding_mask: Optional[Tensor] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        src_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        prompt_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        feat_sizes: Optional[list] = None,
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*

        # ========================================
        memory_mask: Optional[Tensor] = None,  # to support memory mask
        pro_fg=None,  # fg prototypes
        pro_bg=None,  # bg prototypes
        args=None,  # additional parameters
        return_prior=False
        # ========================================
    ):
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = (
                src[0],
                src_key_padding_mask[0],
                src_pos[0],
            )

        assert (
            src.shape[1] == prompt.shape[1]
        ), "Batch size must be the same for src and prompt"

        output = src
        # ========================================
        q = output.clone()
        # ========================================
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            # ========================================
            q = q.transpose(0, 1)
            # ========================================
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)
        
        # ========================================

        # ========================================
        # Cross Attention Bias
        # - if bias_type is not v1 or v2
        # ========================================
        priors_ = None
        h = w = output.size(1) ** 0.5
        fts_size = (int(h), int(w))
        if args.bias_type in ["v3"]:
            assert pro_fg is not None and pro_bg is not None, "When bias_type is not v3, please extract fg and bg prototypes from original features."
            assert memory_mask is not None, "Please ensure memory_mask is available."
            
            
            q = rearrange(output, 'b (h w) c -> b c h w', h=int(h), w=int(w))  # b, c, h, w
            enc = self.pe(q)
            
            if args.bias_mode == "kalman_pos":
                enc_priors = self.generate_prior_enc(enc, enc, [memory_mask[:, -1:, ...]], fts_size=fts_size)[0]
                _, _, priors_1, priors_2 = self.generate_prior_feat_enc_batch(
                    q, pro_fg, pro_bg, enc_priors, fts_size=fts_size
                )  # b, n, h, w
                
                priors_ = [priors_1.mean(1, True), priors_2.mean(1, True)]  # check_results()
            else:
                print("Bias mode should be kalman_pos")
                sys.exit(0)
        else:
            print("Bias type should be v3")
            sys.exit(0)
        
        for idx, layer in enumerate(self.layers):
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                args.layer_idx = idx
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = activation_ckpt_wrapper(layer)(
                tgt=output,
                memory=prompt,  # fg memory
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                **kwds,
            )
            normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        if not return_prior:
            return {
                "memory": normed_output,
                "pos_embed": src_pos,
                "padding_mask": src_key_padding_mask,
            }
        else:
            return {
                "memory": normed_output,
                "pos_embed": src_pos,
                "padding_mask": src_key_padding_mask,
                "priors_": priors_,
            }
        # ========================================
        

# ========================================

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
 
    def generate_prior_enc(self, qry_feat, sup_feat, sup_masks, fts_size):
        bsize, ch_sz, sp_sz, _ = qry_feat.size()[:]
        cos_eps = 1e-7
        
        sim_fgs = []
        for sup_mask in sup_masks:
            # resize mask
            resize_size = sup_feat.size()[-2:]
            sup_mask = F.interpolate(sup_mask, size=resize_size, mode='bilinear', align_corners=False)  # b, 1, h, w

            # fg prototype
            sup_fg = self.Weighted_GAP(sup_feat, sup_mask)
            
            # cosine similarities
            sim_fg = self.cos_sim(qry_feat, sup_fg, cos_eps)
            
            # fg and bg priors, normalize to [0, 1]
            sim_fg = sim_fg.max(1)[0].view(bsize, sp_sz * sp_sz)
            sim_fg = (sim_fg - sim_fg.min(1)[0].unsqueeze(1)) / (
                        sim_fg.max(1)[0].unsqueeze(1) - sim_fg.min(1)[0].unsqueeze(1) + cos_eps)
            sim_fg = sim_fg.view(bsize, 1, sp_sz, sp_sz)

            # interpolate
            sim_fg = F.interpolate(sim_fg, size=fts_size, mode='bilinear', align_corners=False)
            
            sim_fgs.append(sim_fg)
        
        return sim_fgs
 
    def generate_prior_feat_enc_batch(self, qry_feat, sup_fgs, sup_bgs, enc_priors, fts_size, cycle=False):
        '''
        Warning!!!
        Note that this function has already been modified, please refer to that in memory_attention_backup for the original one.
        '''
        bsize, ch_sz, sp_sz, _ = qry_feat.size()[:]
        cos_eps = 1e-7
        
        sup_fg = torch.cat(sup_fgs, dim=2) if isinstance(sup_fgs, list) else sup_fgs  # b, c, n, 1
        sup_bg = torch.cat(sup_bgs, dim=2) if isinstance(sup_bgs, list) else sup_bgs
        enc_prior = torch.cat(enc_priors, dim=1) if isinstance(enc_priors, list) else enc_priors
        num = sup_fg.size(-2)
        
        # ========================================
        # first prior (without scond norm)
        # ========================================
        # cos sim
        sim_fg = self.cos_sim(qry_feat, sup_fg, cos_eps)  # b, n-1, hw
        sim_bg = self.cos_sim(qry_feat, sup_bg, cos_eps)
        
        # # local norm (within each previous frame)
        # sim_fg = (sim_fg - sim_fg.min(-1)[0].unsqueeze(-1)) / (sim_fg.max(-1)[0].unsqueeze(-1) - sim_fg.min(-1)[0].unsqueeze(-1) + cos_eps)
        # sim_bg = (sim_bg - sim_bg.min(-1)[0].unsqueeze(-1)) / (sim_bg.max(-1)[0].unsqueeze(-1) - sim_bg.min(-1)[0].unsqueeze(-1) + cos_eps)
        # # sim_fg = (sim_fg + 1.) / 2.
        # # sim_bg = (sim_bg + 1.) / 2.
        # sim_fg = rearrange(sim_fg, 'b n (h w) -> b n h w', h=sp_sz)
        # sim_bg = rearrange(sim_bg, 'b n (h w) -> b n h w', h=sp_sz)

        # # disc prior - more similar to fg than bg
        # sim_disc = (sim_fg - sim_bg)  # b, n, h, w
 
        # # the only difference lies in the position of using enc_prior (kalman info)
        # if not cycle:
        #     # output 1 - prior without second norm
        #     sim_disc[sim_disc < 0] = 0
        #     sim_disc_1 = sim_disc.clone()
        #     sim_disc_1 = rearrange(sim_disc_1, 'b n h w -> b n (h w)')
        #     sim_disc_1 = (sim_disc_1 - 0) / (sim_disc_1.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        #     sim_disc_1 = rearrange(sim_disc_1, 'b n (h w) -> b n h w', h=sp_sz)
        #     sim_disc_1 = F.interpolate(sim_disc_1, size=fts_size, mode='bilinear', align_corners=False)
    
        #     # ========================================
        #     # second prior (with second norm)
        #     # ========================================
        #     # output 2 - prior with second local norm (within each previous frame)
        #     sim_disc = sim_disc * enc_prior
        #     sim_disc = rearrange(sim_disc, 'b n h w -> b n (h w)')
        #     sim_disc = (sim_disc - 0) / (sim_disc.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        #     sim_disc = rearrange(sim_disc, 'b n (h w) -> b n h w', h=sp_sz)
        #     sim_disc_2 = F.interpolate(sim_disc, size=fts_size, mode='bilinear', align_corners=False)
        # else:
        #     # output 1 - prior without second norm
        #     sim_disc[sim_disc < 0] = 0  # do not use enc_prior at this moment
        #     sim_disc_1 = F.interpolate(sim_disc, size=fts_size, mode='bilinear', align_corners=False)
            
        #     # output 2 - prior with second local norm (within each previous frame)
        #     sim_disc = rearrange(sim_disc, 'b n h w -> b n (h w)')
        #     sim_disc = (sim_disc - 0) / (sim_disc.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        #     sim_disc = rearrange(sim_disc, 'b n (h w) -> b n h w', h=sp_sz)
        #     sim_disc = sim_disc * enc_prior
        #     sim_disc_2 = F.interpolate(sim_disc, size=fts_size, mode='bilinear', align_corners=False)
        
        # local norm (within each previous frame)
        sim_fg_1 = (sim_fg - sim_fg.min(-1)[0].unsqueeze(-1)) / (sim_fg.max(-1)[0].unsqueeze(-1) - sim_fg.min(-1)[0].unsqueeze(-1) + cos_eps)
        sim_bg_1 = (sim_bg - sim_bg.min(-1)[0].unsqueeze(-1)) / (sim_bg.max(-1)[0].unsqueeze(-1) - sim_bg.min(-1)[0].unsqueeze(-1) + cos_eps)
        sim_fg_2 = (sim_fg + 1.) / 2.
        sim_bg_2 = (sim_bg + 1.) / 2.
        
        sim_fg_1 = rearrange(sim_fg_1, 'b n (h w) -> b n h w', h=sp_sz)
        sim_bg_1 = rearrange(sim_bg_1, 'b n (h w) -> b n h w', h=sp_sz)
        sim_fg_2 = rearrange(sim_fg_2, 'b n (h w) -> b n h w', h=sp_sz)
        sim_bg_2 = rearrange(sim_bg_2, 'b n (h w) -> b n h w', h=sp_sz)

        # disc prior - more similar to fg than bg
        sim_disc_1 = (sim_fg_1 - sim_bg_1)  # b, n, h, w
        sim_disc_2 = (sim_fg_2 - sim_bg_2)  # b, n, h, w
 
        # # output 1 - minmax norm
        # sim_disc_1[sim_disc_1 < 0] = 0
        # sim_disc_1 = sim_disc_1 * enc_prior
        # sim_disc_1 = rearrange(sim_disc_1, 'b n h w -> b n (h w)')
        # sim_disc_1 = (sim_disc_1 - 0) / (sim_disc_1.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        # sim_disc_1 = rearrange(sim_disc_1, 'b n (h w) -> b n h w', h=sp_sz)
        # sim_disc_1 = F.interpolate(sim_disc_1, size=fts_size, mode='bilinear', align_corners=False)

        # output 2 - shift norm
        sim_disc_2[sim_disc_2 < 0] = 0
        
        sim_disc_1 = sim_disc_2.clone()
        sim_disc_1 = rearrange(sim_disc_1, 'b n h w -> b n (h w)')
        sim_disc_1 = (sim_disc_1 - 0) / (sim_disc_1.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        sim_disc_1 = rearrange(sim_disc_1, 'b n (h w) -> b n h w', h=sp_sz)
        sim_disc_1 = F.interpolate(sim_disc_1, size=fts_size, mode='bilinear', align_corners=False)
        
        sim_disc_2 = sim_disc_2 * enc_prior
        sim_disc_2 = rearrange(sim_disc_2, 'b n h w -> b n (h w)')
        sim_disc_2 = (sim_disc_2 - 0) / (sim_disc_2.max(-1)[0].unsqueeze(-1) - 0 + cos_eps)
        sim_disc_2 = rearrange(sim_disc_2, 'b n (h w) -> b n h w', h=sp_sz)
        sim_disc_2 = F.interpolate(sim_disc_2, size=fts_size, mode='bilinear', align_corners=False)
            
        # return sim_fg, sim_bg, sim_disc_1, sim_disc_2
        return sim_fg_1, sim_bg_1, sim_disc_1, sim_disc_2

    def Weighted_GAP(self, supp_feat, mask):
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat

    def cos_sim(self, qry_feat, sup_feat, cos_eps=1e-7):
        q = qry_feat.flatten(2).transpose(-2, -1)
        s = sup_feat.flatten(2).transpose(-2, -1)

        qry = q
        qry = qry.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
        qry_norm = torch.norm(qry, 2, 1, True)

        sup = s
        sup = sup.contiguous()
        if sup.size(1) == 1:
            sup_norm = torch.norm(sup, 2, 2, True)
        else:
            sup_norms = []
            for i in range(sup.size(1)):
                sup_norm = torch.norm(sup[:, i:i+1, :], 2, 2, True)
                sup_norms.append(sup_norm)
            sup_norm = torch.cat(sup_norms, dim=1)  # b, n, c

        cos = torch.bmm(sup, qry) / (torch.bmm(sup_norm, qry_norm) + cos_eps)
        return cos

    
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
        tensor = tensor.permute(0, 2, 3, 1)  # b, h, w, c
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)  # b, c, h, w

    @property
    def org_channels(self):
        return self.penc.org_channels
    
# ========================================


class TransformerDecoderLayerv1(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ):
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # Self attention
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        **kwargs,
    ):
        if dac:
            # we only apply self attention to the first half of the queries
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # Recombine
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        **kwds: Any,
    ) -> torch.Tensor:
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            attn_bias=attn_bias,
            **kwds,
        )


class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    def __init__(self, cross_attention_first=False, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        if self.cross_attn_image is None:
            return tgt

        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory, # fg memory
        dac: bool,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ):
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args: Any, **kwds: Any) -> torch.Tensor:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError
