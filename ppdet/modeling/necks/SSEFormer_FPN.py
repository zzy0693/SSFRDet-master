# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

import paddle
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ppdet.modeling.Encoder.SSFE.DAM import DAM
from ppdet.modeling.Encoder.SSFE.CAM import CAM
from ppdet.modeling.Encoder.LKMIAM import LKMIAM


__all__ = ['SSEFormer_FPN']


@register
@serializable
class SSEFormer_FPN(nn.Layer):
    __shared__ = ['eval_size']
    __inject__ = ['encoder_layer']


    def __init__(self,
                 in_channels,
                 out_channel,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 extra_stage=1,
                 use_c5=True,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 eval_size=None,
                 relu_before_extra_convs=True):
        super(SSEFormer_FPN, self).__init__()
        self.out_channel = out_channel
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.num_encoder_layers = num_encoder_layers
        self.use_encoder_idx = use_encoder_idx
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size


        self.LKMIAM = LKMIAM(2048)
        self.res_up2 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=1)
        self.pos_embed = nn.Conv2D(512, 512, 5, padding=2, groups=512)
        self.lateral_convs = []
        self.fpn_convs = []
        fan = out_channel * 3 * 3

        # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
        # 0 <= st_stage < ed_stage <= 3
        st_stage = 4 - len(in_channels)
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i - st_stage]
            if self.norm_type is not None:
                lateral = self.add_sublayer(
                    lateral_name,
                    ConvNormLayer(
                        ch_in=in_c,
                        ch_out=out_channel,
                        filter_size=1,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=in_c)))
            else:
                lateral = self.add_sublayer(
                    lateral_name,
                    nn.Conv2D(
                        in_channels=in_c,
                        out_channels=out_channel,
                        kernel_size=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=in_c))))
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            if self.norm_type is not None:
                fpn_conv = self.add_sublayer(
                    fpn_name,
                    ConvNormLayer(
                        ch_in=out_channel,
                        ch_out=out_channel,
                        filter_size=3,
                        stride=1,
                        norm_type=self.norm_type,
                        norm_decay=self.norm_decay,
                        freeze_norm=self.freeze_norm,
                        initializer=XavierUniform(fan_out=fan)))
            else:
                fpn_conv = self.add_sublayer(
                    fpn_name,
                    nn.Conv2D(
                        in_channels=out_channel,
                        out_channels=out_channel,
                        kernel_size=3,
                        padding=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=fan))))
            self.fpn_convs.append(fpn_conv)

        # add extra conv levels for RetinaNet(use_c5)/FCOS(use_p5)
        if self.has_extra_convs:
            for i in range(self.extra_stage):
                lvl = ed_stage + 1 + i
                if i == 0 and self.use_c5:
                    in_c = in_channels[-1]
                else:
                    in_c = out_channel
                extra_fpn_name = 'fpn_{}'.format(lvl + 2)
                if self.norm_type is not None:
                    extra_fpn_conv = self.add_sublayer(
                        extra_fpn_name,
                        ConvNormLayer(
                            ch_in=in_c,
                            ch_out=out_channel,
                            filter_size=3,
                            stride=2,
                            norm_type=self.norm_type,
                            norm_decay=self.norm_decay,
                            freeze_norm=self.freeze_norm,
                            initializer=XavierUniform(fan_out=fan)))
                else:
                    extra_fpn_conv = self.add_sublayer(
                        extra_fpn_name,
                        nn.Conv2D(
                            in_channels=in_c,
                            out_channels=out_channel,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            weight_attr=ParamAttr(
                                initializer=XavierUniform(fan_out=fan))))
                self.fpn_convs.append(extra_fpn_conv)
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # # encoder transformer
        self.blocks = nn.LayerList([
            DAM(dim=256, num_heads=4,
                        kernel_size=3, )
            for _ in range(len(use_encoder_idx))])

        self.pan_blocks = nn.LayerList([
            CAM(
                256)
            for _ in range(len(use_encoder_idx))])


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, body_feats):
        global c1
        laterals = []
        num_levels = len(body_feats)

        body_feats[2] = self.LKMIAM(body_feats[2])

        for i in range(num_levels):
            laterals.append(self.lateral_convs[i](body_feats[i]))

        aa = F.avg_pool2d(laterals[0], kernel_size=2, stride=4)
        bb = F.avg_pool2d(laterals[1], kernel_size=2, stride=2)
        c = paddle.concat(
            [aa, bb], axis=1)  # fusion
        cc = paddle.concat(
            [laterals[2], bb], axis=1)  # fusion

        c2 = self.res_up2(cc)
        c = self.res_up2(c)

        for idx in range(len(self.use_encoder_idx)):
            c1 = self.pan_blocks[idx](c)#fusion

        if self.num_encoder_layers > 0:  # 插入AIFI
            for i, enc_ind in enumerate(self.use_encoder_idx):
                c0= self.blocks[i](c2)  # 加自注意力
                h, w = c2.shape[2:]  # 提取H,W维度
                src_flatten1= c1.flatten(2).transpose(
                    [0, 2, 1])  # flatten [B, C, H, W] to [B, HxW, C]并且转成序列
                src_flatten = c0.flatten(2).transpose(
                    [0, 2, 1])  # flatten [B, C, H, W] to [B, HxW, C]并且转成序列
                if self.training or self.eval_size is None:  # 获取位置信息
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.out_channel, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten1,src_flatten, pos_embed=pos_embed)  # 加自注意力
                c = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.out_channel, h, w])  # 还原维度


        # FCM
        cc = F.interpolate(
            c,
            scale_factor=2.,
            mode='nearest', )
        ccc = F.interpolate(
            cc,
            scale_factor=2.,
            mode='nearest', )

        c = paddle.concat(
            [c, laterals[2]], axis=1)  # fusion
        c  =self.pos_embed(c)+c
        laterals[2] = self.res_up2(c)

        cc = paddle.concat(
            [cc, laterals[1]], axis=1)  # fusion
        cc  =self.pos_embed(cc)+cc
        laterals[1] = self.res_up2(cc)

        ccc = paddle.concat(
            [ccc, laterals[0]], axis=1)  # fusion
        ccc  =self.pos_embed(ccc)+ccc
        laterals[0] = self.res_up2(ccc)

        for i in range(1, num_levels):
            lvl = num_levels - i
            upsample = F.interpolate(
                laterals[lvl],
                scale_factor=2.,
                mode='nearest', )
            laterals[lvl - 1] += upsample  # 2次上采样

        fpn_output = []
        for lvl in range(num_levels):
            fpn_output.append(self.fpn_convs[lvl](laterals[lvl]))

        if self.extra_stage > 0:
            # use max pool to get more levels on top of outputs (Faster R-CNN, Mask R-CNN)
            if not self.has_extra_convs:
                assert self.extra_stage == 1, 'extra_stage should be 1 if FPN has not extra convs'
                fpn_output.append(F.max_pool2d(fpn_output[-1], 1, stride=2))  # 下采样

            else:
                if self.use_c5:
                    extra_source = body_feats[-1]
                else:
                    extra_source = fpn_output[-1]
                fpn_output.append(self.fpn_convs[num_levels](extra_source))

                for i in range(1, self.extra_stage):
                    if self.relu_before_extra_convs:
                        fpn_output.append(self.fpn_convs[num_levels + i](F.relu(
                            fpn_output[-1])))
                    else:
                        fpn_output.append(self.fpn_convs[num_levels + i](
                            fpn_output[-1]))
        return fpn_output

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]



