import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from fling.utils import get_data_transform
from fling.utils.registry_utils import DATASET_REGISTRY
import random


common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

@DATASET_REGISTRY.register('tiny_imagenet_test')
class TinyImagenetDataset(Dataset):
    r"""
        Implementation for Tiny-Imagenet dataset. Details can be viewed in:
        http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """

    default_augmentation = dict(
        horizontal_flip=dict(p=0.5),
        random_rotation=dict(degree=15),
        Normalize=dict(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    )

    def __init__(self, cfg: dict, train: bool):
        super(TinyImagenetDataset, self).__init__()
        self.train = train
        self.cfg = cfg
        self.common_corruptions = cfg.data.corruption
        transform = get_data_transform(cfg.data.transforms, train=train)
        self._prepare_test_data(train, transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict:
        return {'input': self.dataset[item][0], 'class_id': self.dataset[item][1]}

    def _prepare_test_data(self, train, transform):
        if self.cfg.data.corruption is None or train:
            print('Test on the original test set')
            self.dataset = ImageFolder(os.path.join(self.cfg.data.data_path, 'Tiny-ImageNet-200', 'tiny-imagenet-200'), transform=transform)
        elif self.cfg.data.corruption in common_corruptions:
            self.dataset = ImageFolder(os.path.join(self.cfg.data.data_path, 'Tiny-ImageNet-C', 'Tiny-ImageNet-C', self.cfg.data.corruption, str(self.cfg.data.level)), transform=transform)
        else:
            raise "Don't have this type of data!"

