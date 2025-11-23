# ------------------------------------------------------------------------
# PDVC
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Base Encoder to create multi-level conv features and positional embedding.
"""

import torch
import torch.nn.functional as F
from torch import nn
from misc.detr_utils.misc import NestedTensor
from .position_encoding import PositionEmbeddingSine
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
class GazeguidedRefinement(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        
        self.dim = dim

        # 文本作为查询的投影 
        self.q_proj = nn.Linear(dim, dim)
        # 视觉作为键值的投影 
        self.kv_proj = nn.Linear(dim, 2*dim)
        # 注意力参数 
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads 
        self.scale = self.head_dim ** -0.5

        # 初始化策略 
        self._init_weights()
 
    def _init_weights(self):
        # 查询投影保留文本特征 
        nn.init.eye_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        # 键值投影视觉特征零初始化保护 
        nn.init.zeros_(self.kv_proj.weight[self.dim:])  # 值矩阵零初始化 
        nn.init.xavier_uniform_(self.kv_proj.weight[:self.dim])  # 键矩阵常规初始化 
 
    def forward(self, text_feat, visual_feat):
        """
        text_feat: 待细化文本特征 [B, Lt, C]
        visual_feat: 引导视觉特征 [B, Lv, C]
        (src,src_gaze)
        """
        # print(self.num_heads)

        text_feat = text_feat.transpose(1, 2)
        visual_feat = visual_feat.transpose(1, 2)

        B, Lt, C = text_feat.shape
        Lv = visual_feat.size(1)

        # 文本查询投影
        q = self.q_proj(text_feat).view(B, Lt, self.num_heads, self.head_dim).transpose(1,2)
        # 视觉键值投影
        k, v = self.kv_proj(visual_feat).chunk(2, dim=-1)

        k = k.view(B, Lv, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, Lv, self.num_heads, self.head_dim).transpose(1,2)

        # 注意力计算
        attn = (q @ k.transpose(-2,-1)) * self.scale

        # attn out
        attn_out = attn

        attn = F.softmax(attn, dim=-1)

        # 特征聚合与残差
        refined = (attn @ v).transpose(1,2).reshape(B, Lt, C)

        # residual and norm
        final_feat = refined + text_feat
        final_feat = final_feat.transpose(1, 2)

        return final_feat, attn_out


class BaseEncoder(nn.Module):
    def __init__(self, num_feature_levels, vf_dim, hidden_dim, num_attn_heads):
        super(BaseEncoder, self).__init__()
        self.pos_embed = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim

        if num_feature_levels > 1:
            
            ### visual
            input_proj_list = []
            in_channels = vf_dim
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            for _ in range(num_feature_levels - 1):
                input_proj_list.append(nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            
            ### gaze
            input_proj_list_gaze = []
            in_channels_gaze = vf_dim
            input_proj_list_gaze.append(nn.Sequential(
                nn.Conv1d(in_channels_gaze, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            for _ in range(num_feature_levels - 1):
                input_proj_list_gaze.append(nn.Sequential(
                    nn.Conv1d(in_channels_gaze, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels_gaze = hidden_dim
            self.input_proj_gaze = nn.ModuleList(input_proj_list_gaze)

            ### attn
            vis_gaze_att_list = []
            vis_gaze_att_list.append(GazeguidedRefinement(self.hidden_dim, num_attn_heads))
            for _ in range(num_feature_levels - 1):
                vis_gaze_att_list.append(GazeguidedRefinement(self.hidden_dim, num_attn_heads))
            self.vis_gaze_att_list = nn.ModuleList(vis_gaze_att_list)
            
            
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(vf_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, vf, mask, duration, gz):
        
        # print(vf.shape,gz.shape)

        # vf: (N, L, C), mask: (N, L),  duration: (N)
        vf = vf.transpose(1, 2)  # (N, L, C) --> (N, C, L)
        # gaze
        gz = gz.transpose(1, 2)
        
        vf_nt = NestedTensor(vf, mask, duration)
        pos0 = self.pos_embed(vf_nt)

        srcs = []
        masks = []
        poses = []
        srcs_gz = []
        attn_maps = []

        src0, mask0 = vf_nt.decompose()
        
        # visual
        src = self.input_proj[0](src0)
        # gaze
        src_gaze = self.input_proj_gaze[0](gz)
        # vis gaze attn
        src, attn_out = self.vis_gaze_att_list[0](src,src_gaze)

        attn_maps.append(attn_out)
        srcs_gz.append(src_gaze)
        srcs.append(src)
        masks.append(mask0)
        poses.append(pos0)
        assert mask is not None

        for l in range(1, self.num_feature_levels):
            if l == 1:
                # visual
                src = self.input_proj[l](vf_nt.tensors)
                # gaze
                src_gaze = self.input_proj_gaze[l](gz)
                # vis gaze attn
                src, attn_out = self.vis_gaze_att_list[l](src,src_gaze)

            else:
                #visual 
                src = self.input_proj[l](srcs[-1])
                #gaze
                src_gaze = self.input_proj_gaze[l](srcs_gz[-1])
                # vis gaze attn
                src, attn_out = self.vis_gaze_att_list[l](src,src_gaze)

            m = vf_nt.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-1:]).to(torch.bool)[0]
            pos_l = self.pos_embed(NestedTensor(src, mask, duration)).to(src.dtype)

            # attn map
            attn_maps.append(attn_out)

            srcs.append(src)
            masks.append(mask)
            poses.append(pos_l)
            srcs_gz.append(src_gaze)
            
        return srcs, masks, poses, srcs_gz, attn_maps

def build_gaze_encoder(args):
    gaze_encoder = BaseEncoder(args.num_feature_levels, args.feature_dim, args.hidden_dim, args.num_attn_heads)
    return gaze_encoder
