from robustbench.data import load_cifar100c, load_cifar10c
from robustbench.utils import load_model
import numpy as np
import gdown
import os


x_test, y_tes = load_cifar10c(10000, data_dir = 'data/CIFAR10')
x_test, y_test = load_cifar100c(10000, data_dir = 'data/CIFAR100')

model =  load_model(model_name = 'Hendrycks2020AugMix_ResNeXt', dataset =  'cifar100', threat_model = 'corruptions')
# model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf')
# url = "https://drive.google.com/file/d/1Xy6kVJ8d27RpfE2t8sPBuczHtrS-ZWwP/view?usp=sharing"

id = "1eJu4wBxzeqf4INlOkuBr9Co1NTG4wIVG"
output = "./pretrain/resnet8_cifar10.ckpt"

os.makedirs('./pretrain')
gdown.download(id = id, output = output)

arr = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ])

np.save('4area.npy', arr)
