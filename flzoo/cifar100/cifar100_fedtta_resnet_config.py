from easydict import EasyDict

import datetime
import os

# Get the current date and time
now = datetime.datetime.now()

# Format the datetime to include in the filename
time_fn = now.strftime("%Y_%m_%d_%H_%M_%S")

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
exp_args = dict(
    data=dict(dataset='cifar100_test', data_path='./data/CIFAR100', sample_method=dict(name='iid', train_num=50000, test_num=500),
              corruption=['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'],
              level=[5], class_number=100),
    learn=dict(
        device='cuda:0', local_eps=1, global_eps=1, batch_size=64, optimizer=dict(name='sgd', lr=0.00001, momentum=0.9)
    ),
    model=dict(
        name='cifarresnext',
        class_number=100,
    ),
    client=dict(name='fedpl_client', client_num=20),
    server=dict(name='base_server'),
    group=dict(name='adapt_group', aggregation_method='st',
               aggregation_parameters=dict(
                   name='all',
               )),
    other=dict(test_freq=3, 
               logging_path='./logging/tsa_fedpl_grad_niid_resnext29_lp1_'+time_fn,
               model_path='./pretrain/Hendrycks2020AugMix_ResNeXt.pt',
               partition_path='4area.npy',
               online=True,
               adap_iter=1,
               ttt_batch=10,
               is_continue=True,
               niid=False,
               is_average= True,
               method='adapt',
            #    method = 'bn',
               pre_trained='cifarresnext',
               resume=True,
               time_slide=10,
               st_lr=1e-4,
               st_epoch=100,
               robust_weight=0.5,
               st='both',
               st_head=1,
               loop = 1,
               alpha = 0.9
               ),
    fed=dict(is_TA=True,
             is_GA=True,
             TA_topk=10000),
    method = dict(name = "tsa", #Ffedtsa or ours
                  feat_sim = "feature", #Output or feature
                  public_data = False,
                  random_data = False
                ),
)

exp_args = EasyDict(exp_args)

iid_text = "niid" if exp_args.other.niid else "iid"
file_name = f"{exp_args.method.name}_{exp_args.method.feat_sim}_{exp_args.client.name}_{iid_text}_resnext29_lp_{exp_args.other.loop}_{time_fn}"
exp_args.other.logging_path = os.path.join('logging', exp_args.data.dataset, exp_args.other.method, exp_args.client.name,file_name )

if __name__ == '__main__':
    from fling.pipeline import FedTTA_Pipeline
    FedTTA_Pipeline(exp_args, seed=0)
