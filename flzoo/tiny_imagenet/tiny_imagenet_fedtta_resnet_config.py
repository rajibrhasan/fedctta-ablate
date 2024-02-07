from easydict import EasyDict

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
exp_args = dict(
    data=dict(dataset='tiny_imagenet_test', data_path='./data/', sample_method=dict(name='iid', train_num=50000, test_num=500),
              corruption=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
              level=[5], class_number=200),
    learn=dict(
        device='cuda:0', local_eps=1, global_eps=1, batch_size=128, optimizer=dict(name='sgd', lr=0.00001, momentum=0.9)
    ),
    model=dict(
        name='imagenet_resnet50',
        input_channel=3,
        class_number=200,
    ),
    client=dict(name='fedshot_client', client_num=20),
    server=dict(name='base_server'),
    group=dict(name='ldawa_group', aggregation_method='avg',
               aggregation_parameters=dict(
                   name='all',
               )),
    other=dict(test_freq=3, logging_path='./logging/0130_tiny_imagenet_resnet50_grad_IID_LDAWA_b20_1e5',
               model_path='./pretrain/resnet50_tiny_imagenet.pt',
               partition_path='../4area.npy',
               online=True,
               adap_iter=1,
               ttt_batch=20,

               is_continue=True,
               niid=False,

               is_average=True,
               method='adapt',
               pre_trained='resnet50',
               resume=True,

               time_slide=5,
               st_lr=1e-4,
               st_epoch=100,

               robust_weight=10,
               st='both',
               st_head=1,

               loop=1,
               ),
    fed=dict(is_TA=True,
             is_GA=True,
             TA_topk=10000),
)

exp_args = EasyDict(exp_args)

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline
    FedTTA_Pipeline(exp_args, seed=228)
