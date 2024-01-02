from easydict import EasyDict

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
exp_args = dict(
    data=dict(dataset='cifar10_test', data_path='./data/CIFAR10', sample_method=dict(name='iid', train_num=50000, test_num=500),
              corruption=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
              level=[5], class_number=10),
    learn=dict(
        device='cuda:0', local_eps=1, global_eps=1, batch_size=64, optimizer=dict(name='sgd', lr=0.001, momentum=0.9)
    ),
    model=dict(
        name='resnet8',
        input_channel=3,
        class_number=10,
    ),
    client=dict(name='fedtta_client', client_num=20),
    server=dict(name='base_server'),
    group=dict(name='adapt_group', aggregation_method='wotime',
               aggregation_parameters=dict(
                   name='all',

               )),
    other=dict(test_freq=3, logging_path='./logging/1228_cifar10_resnet_STiid_cotta_st',
               model_path='./pretrain/resnet8_cifar10.ckpt',
               online=True,
               adap_iter=1,
               ttt_batch=10,

               is_continue=False,
               niid=False,

               is_average=True,
               method='bn',
               pre_trained='resnet8',
               resume=True,
               ),
    fed=dict(is_TA=True,
             is_GA=True,
             TA_topk=10000),
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline
    FedTTA_Pipeline(exp_args, seed=0)
