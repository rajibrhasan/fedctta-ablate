from easydict import EasyDict

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
exp_args = dict(
    data=dict(dataset='cifar10_test', data_path='./data/CIFAR10/', sample_method=dict(name='iid', train_num=50000, test_num=10000),
              corruption=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
              level=[5], class_number=10),
    learn=dict(
        device='cuda:0', local_eps=1, global_eps=1, batch_size=200, optimizer=dict(name='sgd', lr=0.001, momentum=0.9)
    ),
    model=dict(
        name='wide_resnet',
        num_classes=10,
        depth=26,
        widen_factor=1,
    ),
    client=dict(name='tta_client', client_num=5),
    server=dict(name='base_server'),
    group=dict(name='adapt_group', aggregation_method='avg',
               aggregation_parameters=dict(
                   name='include',
                   keywords='block1'
               )),
    other=dict(test_freq=3, logging_path='./logging/1204_cifar10_wideresnet',
               model_path='./pretrain/natural.pt.tar',
               online=True,
               adap_iter=1,
               ttt_batch=8,
               is_continue=False,
               is_average=False,
               method='fed',
               niid=False,
               pre_trained='wideresnet28',
               ),
    fed=dict(is_TA=True,
             is_GA=True,
             TA_topk=10000),
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline

    FedTTA_Pipeline(exp_args, seed=0)