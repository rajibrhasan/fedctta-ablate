import os.path

import tqdm
import random
import numpy as np
import torch
import torch.nn as nn

from fling.component.client import get_client
from fling.component.server import get_server
from fling.component.group import get_group
from fling.dataset import get_dataset
from fling.utils.data_utils import data_sampling
from fling.utils import Logger, compile_config, client_sampling, VariableMonitor, LRScheduler

from fling.model import get_model
from torch.utils.data import DataLoader
from fling.utils.data_utils.sampling import NaiveDataset

import pandas as pd
from fling.utils.utils import VariableMonitor, SaveEmb
import pickle
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import datetime

def preprocess_data(data, device):
    return {'x': data['input'].to(device), 'y': data['class_id'].to(device)}


def non_iid_continual(args, is_niid, client_number, corupt_number):
    if not is_niid:
        corupt_map = np.array([[i for i in range(corupt_number)] for _ in range(client_number)])
    else:
        corupt_map = np.load(args.other.partition_path)
    return corupt_map

def test_origin(net, test_dataloader):
    # Without Corupt Test
    monitor = VariableMonitor()
    net.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            preprocessed_data = {'x': data['input'].cuda(), 'y': data['class_id'].cuda()}
            batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

            out = net(batch_x)
            y_pred = torch.argmax(out, dim=-1)
            loss = criterion(out, batch_y)
            monitor.append(
                {
                    'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                    'test_loss': loss.item()
                },
                weight=preprocessed_data['y'].shape[0]
            )
    mean_monitor_variables = monitor.variable_mean()
    print(f'Origin test acc {mean_monitor_variables["test_acc"]}, test loss {mean_monitor_variables["test_loss"]}')

def init_tta_state(args, net, ckpt, logger, corrupt_dict, corrupt_test_sets, origin_test_sets, trainloader):

    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=corrupt_dict)
    for i in range(args.client.client_num):
        group.append(
            get_client(args=args, client_id=i, train_dataset=corrupt_test_sets[i], test_dataset=origin_test_sets[i]))
        group.clients[i].init_weight(ckpt=ckpt)
    group.initialize()

    if args.client.name == 'fedthe_client':
        global_rep = init_train_feature(args, net, trainloader)
        for i in range(args.client.client_num):
            group.clients[i].init_globalrep(global_rep)
    elif args.client.name == 'fedactmad_client':
        clean_feature_mean, clean_feature_var = init_train_bn(net, trainloader)
        for i in range(args.client.client_num):
            group.clients[i].update_statistics(clean_feature_mean, clean_feature_var)

    return group

def init_train_bn(net, trainloader):
    chosen_layers = []
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            chosen_layers.append(m)

    n_chosen_layers = len(chosen_layers)
    hook_list = [SaveEmb() for _ in range(n_chosen_layers)]
    clean_feature_mean = []
    clean_feature_var = []
    for idx, data in enumerate(trainloader):
        hooks = [chosen_layers[i].register_forward_hook(hook_list[i]) for i in range(n_chosen_layers)]
        inputs = data['input'].cuda()
        with torch.no_grad():
            net.eval()
            _ = net(inputs)
            for yy in range(n_chosen_layers):
                hook_list[yy].statistics_update(), hook_list[yy].clear(), hooks[yy].remove()

    for i in range(n_chosen_layers):
        clean_feature_mean.append(hook_list[i].pop_mean()), clean_feature_var.append(hook_list[i].pop_var())
    return clean_feature_mean, clean_feature_var

def init_train_feature(args, net, trainloader):
    feature_bank = []
    with torch.no_grad():
        for idx, data in enumerate(trainloader):
            x, y = data['input'].to(args.learn.device), data['class_id'].to(args.learn.device)
            feature, out = net(x, mode='compute-feature-logit')
            if feature_bank == []:
                feature_bank = feature
            else:
                feature_bank = torch.concatenate((feature_bank, feature), dim=0)
    feature_mean = torch.mean(feature_bank, dim=0)
    return feature_mean


    
def Train_Model(args: dict, seed: int = 0) -> None:
    # Compile the input arguments first.
    args = compile_config(args, seed)
    # Construct logger.
    logger = Logger(args.other.logging_path)
    wandb.login(key="f22f65f9d8ca626c5d8a36d80eee6506d14501c3")
    
    wandb.init(project="resnet_train", dir="output", name=args.other.logging_path.split('/')[-1])

    # Load origin train dataset `origin_train_set`, origin test dataset `origin_test_set`
    dataset_name = args.data.dataset
    pos = args.data.dataset.rfind('_')
    if pos != -1:
        args.data.dataset = args.data.dataset[:pos]
    print(dataset_name, args.data.dataset)
    origin_test_set = get_dataset(args, train=False)
    origin_train_set = get_dataset(args, train=True)
    train_dataloader = DataLoader(origin_train_set, batch_size=args.learn.batch_size, shuffle=True)
    test_dataloader = DataLoader(origin_test_set, batch_size=args.learn.batch_size, shuffle=False)

    args.data.dataset = dataset_name
    net = get_model(args)
    net.cuda()
    test_origin(net, test_dataloader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)

    for epoch in tqdm.tqdm(range(args.learn.global_eps)):
        net.train()
        for _, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            preprocessed_data = preprocess_data(data, device= args.learn.device)
            batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
            outputs = net(batch_x)
            y_pred = torch.argmax(outputs, dim=-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()


        test_origin(net, test_dataloader)
    
    torch.save(net.state_dict(), 'pretrain/resnet8_cifar10.ckpt')
   