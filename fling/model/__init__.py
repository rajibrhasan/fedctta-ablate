# General models
from .gpt import GPT
from .mlp import MLPModel
from .cnn import CNNModel
from .resnet import resnet4, resnet6, resnet8, resnet10, resnet18, resnet34, resnet50
from .swin_transformer import SwinTransformer
from .vit import ViT
from .wideresnet import WideResNet
from .resnext import CifarResNeXt
from .build_model import get_model
from .imagenet_resnet import imagenet_resnet18, imagenet_resnet50
from .cifar100_resnet import ResNet
from .cifar100_wideresnet_origin import DMWideResNet
# Algorithm specific models
# FedRoD
from .fedrod_resnet import fedrod_resnet4, fedrod_resnet6, fedrod_resnet8, fedrod_resnet10, fedrod_resnet18,\
    fedrod_resnet34, fedrod_resnet50
