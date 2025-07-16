import torch.nn as nn
import torch
from mmcv.cnn import normal_init, kaiming_init, xavier_init, ConvModule
from mmcv.runner import BaseModule
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmocr.models.textdet.postprocess import lra_decode
from mmocr.models.textdet.dense_heads.head_mixin import HeadMixin
from ..postprocess.lra_decoder import  poly_nms
import math
import numpy as np

@HEADS.register_module()
class LRAHeadKACSingle(HeadMixin, BaseModule):

    def __init__(self,
                 in_channels,
                 scales,
                 num_coefficients,
                 path_lra,
                 loss=dict(type='LRALoss'),
                 score_thr=0.1,
                 nms_thr=0.1,
                 num_convs=0,
                 box_iou=False,
                 train_cfg=None,
                 test_cfg=None,
                 n_kernels=3,
                 kac_included=True,
                 is_efficient=False):

        super().__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        loss['steps'] = scales
        self.loss_module = build_loss(loss)
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_convs = num_convs
        self.out_channels_reg = num_coefficients
        self.box_iou = box_iou
        U_t = np.load(path_lra)['components_c']
        U_t = torch.from_numpy(U_t)
        self.U_t = U_t
        self.n_kernels = n_kernels

        self.kac_included = kac_included
        if self.kac_included and is_efficient:
            self.kac = EfficientKACBlock(in_channels, n_kernels=self.n_kernels)
        elif self.kac_included and not is_efficient:
            self.kac = KACBlock(in_channels, n_kernels=self.n_kernels)

        
        if self.num_convs > 0:
            cls_convs = []
            reg_convs = []
            conv_cfg = None
            norm = None
            for i in range(self.num_convs):

                cls_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                                            conv_cfg=conv_cfg if i < 3 else None, norm_cfg=norm, act_cfg=dict(type='ReLU')))
                reg_convs.append(ConvModule(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1,
                                            conv_cfg=conv_cfg if i < 3 else None, norm_cfg=norm, act_cfg=dict(type='ReLU')))
            self.cls_convs = nn.Sequential(*cls_convs)
            self.reg_convs = nn.Sequential(*reg_convs)

        self.out_conv_cls_dense = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1)
        self.out_conv_reg_dense = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

        self.out_conv_cls_sparse = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1)

        self.out_conv_reg_sparse = nn.Conv2d(
            self.in_channels,
            self.out_channels_reg,
            kernel_size=3,
            stride=1,
            padding=1)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.out_conv_cls_sparse.bias, bias_value)
        self.init_weights()

    def init_weights(self):
        normal_init(self.out_conv_cls_dense, mean=0, std=0.01)
        normal_init(self.out_conv_reg_dense, mean=0, std=0.01)
        normal_init(self.out_conv_reg_sparse, mean=0, std=0.01)

    def forward(self, feats):
        feat = feats[0]
        cls_dense, reg_dense, cls_sparse, reg_sparse = self.forward_single(feat)
        preds = [[cls_dense, reg_dense, cls_sparse, reg_sparse]]
        return preds

    def forward_single(self, x):
        if self.kac_included:
            x = self.kac(x)
        if self.num_convs > 0:
            x_cls = self.cls_convs(x)
            x_reg = self.reg_convs(x)
        else:
            x_cls = x
            x_reg = x
        cls_predict_dense = self.out_conv_cls_dense(x_cls)
        reg_predict_dense = self.out_conv_reg_dense(x_reg)
        cls_predict_sparse = self.out_conv_cls_sparse(x_cls)
        reg_predict_sparse = self.out_conv_reg_sparse(x_reg)

        return cls_predict_dense, reg_predict_dense, cls_predict_sparse, reg_predict_sparse


    def get_boundary(self, score_maps, img_metas, rescale):

        assert len(score_maps) == len(self.scales)

        boundaries = []

        for idx, score_map in enumerate(score_maps):

            scale = self.scales[idx]
            boundary = self._get_boundary_single(self.U_t.cuda(), score_map, scale)
            boundaries = boundaries + boundary

        boundaries, _ = poly_nms(boundaries, self.nms_thr, with_index=True)

        if rescale:
            boundaries = self.resize_boundary(
                boundaries, 1.0 / img_metas[0]['scale_factor'])

        results = dict(boundary_result=boundaries, scales=self.scales)
        return results

    def _get_boundary_single(self, U_t, score_map, scale):

        return lra_decode(
            U_t = U_t, 
            preds=score_map,
            scale=scale,
            score_thr=self.score_thr,
        )


class KACBlock(nn.Module):
    def __init__(self, in_channels, n_kernels=3):
        super(KACBlock, self).__init__()
        self.in_channels = in_channels
        self.n_kernels = n_kernels

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=2*(i-1)+1, padding=(2*(i-1)+1)//2) for i in range(1, n_kernels+1)
                ])
        
        self.adaptive_weight = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=5, padding=5//2),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, n_kernels, kernel_size=3, padding=3//2),
            nn.Sigmoid()
        )

        # 1x1 convolution to fuse the outputs of the KACL module
        # This is to reduce the number of channels from in_channels*n_kernels to in_channels
        self.fuse_conv = nn.Conv2d(in_channels*n_kernels, in_channels, kernel_size=1)
        # Batch normalization
        self.bn = nn.BatchNorm2d(in_channels)
        # ReLU activation
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        # KACW part
        # weights shape: (batch_size, n_kernels, height, width)
        weights = self.adaptive_weight(x)
        
        weights = weights / (weights.sum(dim=1, keepdim=True)+1e-6) # Normalize the weights according to the paper

        # KACL part
        # New version
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            weighted_out = conv(x) * weights[:, i:i+1]
            conv_outputs.append(weighted_out)
        
        stacked = torch.cat(conv_outputs, dim=1) # shape: (batch_size, in_channels*n_kernels, height, width)
        # Apply the 1x1 convolution to fuse the outputs
        out = self.fuse_conv(stacked)
        
  


        # Concatenate the conv_outs along the kernel dimension
        # out shape: (batch_size, in_channels*n_kernels, height, width)
        #out = stacked_convs.view(stacked_convs.size(0), -1, stacked_convs.size(3), stacked_convs.size(4))  # (B, C*n_kernels, H, W)

        # out shape [B, in_channels, H, W]
        out = self.bn(out)
        out = self.relu(out)
        # out shape [B, in_channels, H, W]
        # Add residual connection
        out = out + x
        # Check NaN
        return out # shape: [batch_size, in_channels, height, width]

    def init_weights(self):
        # For the kernel weights, we use kaiming initialization
        for conv in self.convs:
            kaiming_init(conv, mode='fan_out', nonlinearity='relu')

        # For the adaptive weights we use kaiming for the first conv and xavier for the second conv
        kaiming_init(self.adaptive_weight[0], mode='fan_out', nonlinearity='relu')
        xavier_init(self.adaptive_weight[2], gain=1, distribution='uniform')


class EfficientKACBlock(nn.Module):
    def __init__(self, in_channels, n_kernels=3, reduction=4):
        super(EfficientKACBlock, self).__init__()
        self.in_channels = in_channels
        self.n_kernels = n_kernels

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=2*(i-1)+1, padding=(2*(i-1)+1)//2)
            for i in range(1, n_kernels+1)
        ])

        self.reduction = reduction  # Reduction factor for the adaptive weight module
        reduced = max(1, in_channels // self.reduction)
        self.adaptive_weight = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, reduced, kernel_size=3, padding=1, groups=reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, n_kernels, kernel_size=1),
            nn.Sigmoid()
        )

        # 1x1 convolution to fuse the outputs of the KACL module
        # This is to reduce the number of channels from in_channels*n_kernels to in_channels
        self.fuse_conv = nn.Conv2d(in_channels*n_kernels, in_channels, kernel_size=1)

        # Batch normalization and ReLU activation
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def forward(self, x):
        weights = self.adaptive_weight(x)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        conv_outputs = [
            conv(x) * weights[:, i:i+1] for i, conv in enumerate(self.convs)
        ]

        out = torch.cat(conv_outputs, dim=1)
        out = self.fuse_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out + x
    
    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv, mode='fan_out', nonlinearity='relu')
        
        # Adaptive weight path
        xavier_init(self.adaptive_weight[0], gain=1.0, distribution='uniform')   # pointwise
        kaiming_init(self.adaptive_weight[2], mode='fan_out', nonlinearity='relu')  # depthwise
        xavier_init(self.adaptive_weight[4], gain=1.0, distribution='uniform')   # pointwise

        # Fuse + BN
        kaiming_init(self.fuse_conv, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
