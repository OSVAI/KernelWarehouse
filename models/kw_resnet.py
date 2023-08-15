import torch
import torch.nn as nn
from modules.kernel_warehouse import Warehouse_Manager
from timm.models.layers import DropPath
from timm.models.registry import register_model


def kwconv3x3(in_planes, out_planes, stride=1, warehouse_name=None, warehouse_manager=None, enabled=True):
    return warehouse_manager.reserve(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                     warehouse_name=warehouse_name, enabled=enabled, bias=False)


def kwconv1x1(in_planes, out_planes, stride=1, warehouse_name=None, warehouse_manager=None, enabled=True):
    return warehouse_manager.reserve(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                                     warehouse_name=warehouse_name, enabled=enabled, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 stage_idx=None, layer_idx=None, warehouse_manager=None, warehouse_handover=False, drop_path=0.):
        super(BasicBlock, self).__init__()
        conv1_stage_idx = max(stage_idx - 1 if warehouse_handover else stage_idx, 0)
        self.conv1 = kwconv3x3(inplanes, planes, stride,
                               warehouse_name='stage{}_layer{}_conv{}'.format(conv1_stage_idx, layer_idx, 0),
                               warehouse_manager=warehouse_manager)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        layer_idx = 0 if warehouse_handover else layer_idx
        self.conv2 = kwconv3x3(planes, planes,
                               warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx, layer_idx, 1),
                               warehouse_manager=warehouse_manager)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.drop_path(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 stage_idx=None, layer_idx=None, warehouse_manager=None, warehouse_handover=False, drop_path=0.):
        super(Bottleneck, self).__init__()
        conv1_stage_idx = stage_idx - 1 if warehouse_handover else stage_idx
        self.conv1 = kwconv1x1(inplanes, planes,
                               warehouse_name='stage{}_layer{}_conv{}'.format(conv1_stage_idx, layer_idx, 0),
                               warehouse_manager=warehouse_manager, enabled=(conv1_stage_idx >= 0))
        self.bn1 = nn.BatchNorm2d(planes)
        layer_idx = 0 if warehouse_handover else layer_idx
        self.conv2 = kwconv3x3(planes, planes, stride,
                               warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx, layer_idx, 1),
                               warehouse_manager=warehouse_manager)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = kwconv1x1(planes, planes * self.expansion,
                               warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx, layer_idx, 2),
                               warehouse_manager=warehouse_manager)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + self.drop_path(out)
        out = self.relu(out)
        return out

@register_model
class KW_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dropout=0.1, reduction=0.0625,
                 cell_num_ratio=1, cell_inplane_ratio=1, cell_outplane_ratio=1,
                 sharing_range=('layer', 'conv'), nonlocal_basis_ratio=1, drop_path_rate=0., **kwargs):
        super(KW_ResNet, self).__init__()
        self.warehouse_manager = Warehouse_Manager(reduction, cell_num_ratio, cell_inplane_ratio, cell_outplane_ratio,
                                                   sharing_range, nonlocal_basis_ratio)
        self.inplanes = 64
        self.layer_idx = 0
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stage_idx=0, warehouse_manager=self.warehouse_manager, drop_path=drop_path_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       stage_idx=1, warehouse_manager=self.warehouse_manager, drop_path=drop_path_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       stage_idx=2, warehouse_manager=self.warehouse_manager, drop_path=drop_path_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       stage_idx=3, warehouse_manager=self.warehouse_manager, drop_path=drop_path_rate)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.warehouse_manager.store()
        self.warehouse_manager.allocate(self)

    def _make_layer(self, block, planes, blocks, stride=1, stage_idx=-1, warehouse_manager=None, drop_path=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                warehouse_manager.reserve(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0,
                    warehouse_name='stage{}_layer{}_conv{}'.format(stage_idx - 1, self.layer_idx + 1, 0),
                    enabled=(stride != 1), bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, stage_idx=stage_idx, layer_idx=self.layer_idx,
                            warehouse_manager=warehouse_manager, warehouse_handover=True, drop_path=drop_path))
        self.layer_idx = 1
        self.inplanes = planes * block.expansion
        for idx in range(1, blocks):
            layers.append(block(self.inplanes, planes, stage_idx=stage_idx, layer_idx=self.layer_idx,
                                warehouse_manager=warehouse_manager, drop_path=drop_path))
            self.layer_idx += 1
        return nn.Sequential(*layers)

    def net_update_temperature(self, temp):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temp)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


@register_model
def kw_resnet18(**kwargs):
    model = KW_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

@register_model
def kw_resnet50(**kwargs):
    model = KW_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

