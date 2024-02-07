from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .mnist import MNISTDataset
from .tiny_imagenet import TinyImagenetDataset
from .mini_imagenet import MiniImagenetDataset
from .imagenet import ImagenetDataset
from .cifar10_test import CIFAR10_TestDataset
from .cifar100_test import CIFAR100_TestDataset
from .tiny_imagenet_test import TinyImagenetDataset

from .build_dataset import get_dataset
