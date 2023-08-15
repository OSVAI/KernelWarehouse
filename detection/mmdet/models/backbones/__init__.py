from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .trident_resnet import TridentResNet
from .swin_transformer import SwinTransformer
from .resnet import ResNet
from .kw_resnet import KW_ResNet
from .convnext import ConvNeXt
from .kw_convnext import KW_ConvNeXt
from .mobilenetv2 import MobileNetV2
from .kw_mobilenetv2 import KW_MobileNetV2

__all__ = [
    'RegNet', 'ResNet', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet', 'DetectoRS_ResNet', 'DetectoRS_ResNeXt', 'Darknet',
    'TridentResNet', 'SwinTransformer', 'KW_ResNet', 'ConvNeXt', 'KW_ConvNeXt', 'MobileNetV2', 'KW_MobileNetV2'
]