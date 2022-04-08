from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # SE module
        # self.se = SELayer(channel=planes, reduction=16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        # SE module
        # self.se = SELayer(channel=planes * self.expansion, reduction=16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class BoundaryConstrainedModule(nn.Module):
    def __init__(self, in_channels, in_channels_2, reduction=4): # in_channels => low_feature  out_channels => high_feature
        super(BoundaryConstrainedModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels_2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels_2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, BCM_input, feature):
        b, c, _, _ = feature.size()
        feat = self.avg_pool(feature).view(b, c)
        feat = self.fc(feat).view(b, self.out_channels, 1, 1)
        BCM_input = BCM_input * feat.expand_as(BCM_input)

        return torch.cat([feature, BCM_input], dim=1)

class ContextEnhancedModule(nn.Module):
    def __init__(self, in_channels, mid_channels, classes):
        super(ContextEnhancedModule, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.classes = classes

        ######################################################
        ########## Context Feature Representation ############
        ######################################################
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.classes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.classes),
            nn.ReLU(inplace=True),
        )
        self.f_gate = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        #######################################################
        ############ Context Attention Extract ################
        #######################################################
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )
        #####################################################
        ########## Context Boost Representation #############
        #####################################################
        self.f_back = nn.Sequential(
            nn.Conv2d(in_channels=self.mid_channels*2, out_channels=self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        # Contextual Feature Representation
        rough_seg = self.f_down(x).view(batch_size, self.classes, -1).permute(0, 2, 1) # [b, h*w, c]
        # rough_seg = F.softmax(rough_seg, dim=2)
        feature = self.f_gate(x)  # [b, k, h, w]
        CFR = torch.matmul(feature.view(batch_size, self.mid_channels, -1), rough_seg).unsqueeze(3) # [b, k, c, 1]

        # Semantic Attention Extraction
        query = self.f_query(x).view(batch_size, self.mid_channels, -1).permute(0, 2, 1)  # [b, h*w, k]
        key = self.f_key(CFR).view(batch_size, self.mid_channels, -1) # [b, k, c]
        value = self.f_value(CFR).view(batch_size, self.mid_channels, -1).permute(0, 2, 1) # [b, c, k]
        sim_map = torch.matmul(query, key) # [b, h*w, c]
        sim_map = (self.mid_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous() # [b, k, h*w]
        context = context.view(batch_size, self.mid_channels, *x.size()[2:]) # [b, k, h, w]

        # Contextual Enhancement Representation
        context = torch.cat([feature, context], dim=1)  # [b, 2*k, h, w]
        context = self.f_back(context) # [b, k, h, w]

        return context


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config):
        extra = config["MODEL"]["EXTRA"]
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int32(np.sum(pre_stage_channels))

        # Boundary Constrain Module
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=num_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.gate_edge_1 = BoundaryConstrainedModule(in_channels=num_channels[0], in_channels_2=num_channels[0])
        self.gate_edge_2 = BoundaryConstrainedModule(in_channels=num_channels[0], in_channels_2=num_channels[0]*2)
        self.gate_edge_3 = BoundaryConstrainedModule(in_channels=num_channels[0], in_channels_2=num_channels[0]*3)
        self.last_edge = nn.Sequential(
            nn.Conv2d(in_channels=num_channels[0]*4, out_channels=num_channels[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channels[0], out_channels=2, kernel_size=1, stride=1, padding=0, bias=False),
        )

        last_inp_channels = np.int32(np.sum(pre_stage_channels)) + num_channels[0]*4

        # Context Enhanced Module
        self.mid_channels = last_inp_channels//4
        self.in_channels = last_inp_channels
        self.mid_channels = 256
        self.CEM = ContextEnhancedModule(in_channels=self.in_channels, mid_channels=self.mid_channels, classes=6)
        last_inp_channels = self.mid_channels

        self.last_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=last_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels=last_inp_channels,
                    out_channels=extra['NUM_CLASSES'],
                    kernel_size=extra['FINAL_CONV_KERNEL'],
                    stride=1,
                    padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def base_forward(self, x):
        input = x
        feature_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        feature_list.append(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        feature_list.append(y_list[0])

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        feature_list.append(y_list[0])

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        feature_list.append(x[0])

        # Boundary Extraction Branch
        edge = self.down(feature_list[0])
        edge = self.gate_edge_1(edge, feature_list[1])
        edge = self.gate_edge_2(edge, feature_list[2])
        edge = self.gate_edge_3(edge, feature_list[3])

        # Image Extraction Branch
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3, edge], 1)
        # Output Boundary result
        edge = self.last_edge(edge)
        edge = F.interpolate(edge, size=input.size()[2:], mode='bilinear', align_corners=True)
        # Output Image result
        x = self.CEM(x)
        x = self.last_layer(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x, edge

    def forward(self, input, tta=False):
        if not tta:
            return self.base_forward(input)
        else:
            out1 = self.base_forward(input)[0]
            out1 = F.softmax(out1, dim=1)
            origin_x1 = input.clone()

            x1 = origin_x1.flip(2)
            cur_out1 = self.base_forward(x1)[0]
            out1 += F.softmax(cur_out1, dim=1).flip(2)

            x1 = origin_x1.flip(3)
            cur_out1 = self.base_forward(x1)[0]
            out1 += F.softmax(cur_out1, dim=1).flip(3)

            x1 = origin_x1.transpose(2, 3).flip(3)
            cur_out1 = self.base_forward(x1)[0]
            out1 += F.softmax(cur_out1, dim=1).flip(3).transpose(2, 3)

            x1 = origin_x1.flip(3).transpose(2, 3)
            cur_out1 = self.base_forward(x1)[0]
            out1 += F.softmax(cur_out1, dim=1).transpose(2, 3).flip(3)

            x1 = origin_x1.flip(2).flip(3)
            cur_out1 = self.base_forward(x1)[0]
            out1 += F.softmax(cur_out1, dim=1).flip(3).flip(2)

            out1 /= 6.0

            return out1

    def init_weights(self, pretrained='', ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            raise NotImplementedError


def HBCNet(backbone, pretrained=False):
    cfg = yaml.load(open("./configs/%s.yaml" % backbone), Loader=yaml.FullLoader)
    if not pretrained:
        cfg["MODEL"]["PRETRAINED"] = ""

    model = HighResolutionNet(cfg)
    model.init_weights(cfg["MODEL"]["PRETRAINED"])

    return model

if __name__ == "__main__":
    net = HBCNet(backbone='hrnet_w40', pretrained=False)
    img = torch.rand(1, 3, 512, 512)
    print(net)
    res = net(img, tta=True)
    print(res.shape)
