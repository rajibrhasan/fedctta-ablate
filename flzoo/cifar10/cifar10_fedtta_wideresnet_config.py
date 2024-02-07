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
        name='cifar10_wideresnet',
    ),
    client=dict(name='fedtta_client', client_num=20),
    server=dict(name='base_server'),
    group=dict(name='adapt_group', aggregation_method='avg',
               aggregation_parameters=dict(
                   name='include',
                   keywords='block1'
               )),
    other=dict(test_freq=3, logging_path='./logging/1204_cifar10_wideresnet',
               model_path='./pretrain/Wang2023Better_wrn-28-10.pt',
               partition_path='../4area.npy',
               online=True,
               adap_iter=1,
               ttt_batch=10,

               is_continue=True,
               niid=True,

               is_average=True,
               method='bn',
               pre_trained='resnet8',
               resume=True,

               time_slide=10,
               st_lr=1e-4,
               st_epoch=100,
               robust_weight=0.5,
               st='both',

               st_head=1,
               loop=5,
               ),

    # fed=dict(is_TA=True,
    #          is_GA=True,
    #          TA_topk=10000),
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline

    FedTTA_Pipeline(exp_args, seed=0)