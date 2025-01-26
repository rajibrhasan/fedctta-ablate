from robustbench.data import load_cifar100c

# x_test, y_test = load_cifar100c(10000, data_dir='data/CIFAR100')

from robustbench.utils import load_model

model =  load_model('Hendrycks2020AugMix_ResNeXt', 'pretrain', 'cifar100', 'corruptions')