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


def non_iid_continual(is_niid, client_number, corupt_number):
    if not is_niid:
        corupt_map = np.array([[i for i in range(corupt_number)] for _ in range(client_number)])
    else:
        # corupt_map = np.array([[-1, ] * corupt_number] * client_number)
        # for j in range(client_number):
        #     for i in range(corupt_number):
        #         temp = random.randint(0, corupt_number - 1)
        #         while temp in corupt_map[j]:
        #             temp = (temp + 1) % corupt_number
        #         corupt_map[j][i] = temp
        corupt_map = np.array([
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

def init_tta_state(args, net, train_dataloader, logger, corrupt_dict, corrupt_test_sets, origin_test_sets):

    group = get_group(args, logger)
    group.server = get_server(args, test_dataset=corrupt_dict)
    for i in range(args.client.client_num):
        group.append(
            get_client(args=args, client_id=i, train_dataset=corrupt_test_sets[i], test_dataset=origin_test_sets[i]))
    group.initialize()

    # initial
    if args.other.method == 'fed':
        clean_mean, clean_var = [], []
        for nm, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                clean_mean.append(m.weighted_mean)
                clean_var.append(m.weighted_var)
        group.init_bnstatistics(clean_mean, clean_var)

    return group


def FedTTA_Pipeline(args: dict, seed: int = 0) -> None:
    # Compile the input arguments first.
    args = compile_config(args, seed)

    # Construct logger.
    logger = Logger(args.other.logging_path)

    # Load origin train dataset `origin_train_set`, origin test dataset `origin_test_set`
    dataset_name = args.data.dataset
    args.data.dataset = args.data.dataset.split('_')[0]
    print(dataset_name, args.data.dataset)
    origin_test_set = get_dataset(args, train=False)
    origin_train_set = get_dataset(args, train=True)
    train_dataloader = DataLoader(origin_train_set, batch_size=args.learn.batch_size, shuffle=True)
    test_dataloader = DataLoader(origin_test_set, batch_size=args.learn.batch_size, shuffle=False)

    # Corrupted Type & Level Test Data
    '''
    corrupt_dict[ corrupt_type ][ corrupt_severity ]
    '''
    args.data.dataset = dataset_name
    corrupt_dict = {}
    common_corrupt = args.data.corruption
    common_serverity = args.data.level
    for corrupt in common_corrupt:
        corrupt_dict[corrupt] = {}
        for serv in common_serverity:
            args.data.corruption = corrupt
            args.data.level = serv
            test_set = get_dataset(args, train=False)
            corrupt_dict[corrupt][serv] = test_set
    args.data.corruption = common_corrupt
    args.data.level = common_serverity

    # Split dataset into clients.
    origin_test_sets = data_sampling(origin_test_set, args, seed, train=False)
    '''
    corrupt_test_sets[ client_id ][ corrupt_type ][ corrupt_severity ]
    '''
    corrupt_test_sets = [{} for _ in range(args.client.client_num)]
    for corrupt in common_corrupt:
        for cidx in range(args.client.client_num):
            corrupt_test_sets[cidx][corrupt] = {}
        for serv in common_serverity:
            test_sets = data_sampling(corrupt_dict[corrupt][serv], args, seed, train=False)
            for cidx in range(args.client.client_num):
                corrupt_test_sets[cidx][corrupt][serv] = test_sets[cidx]

    # load pre-trained net
    ckpt = torch.load(args.other.model_path)
    net = get_model(args)
    print(net)
    if args.other.pre_trained == 'wideresnet28' and args.data.class_number == 10:
        ckpt = ckpt['state_dict']
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'wideresnet28' and args.data.class_number == 100:
        ckpt = ckpt['model_state_dict']
        net.load_state_dict(ckpt)
    elif args.other.pre_trained == 'cifarresnext' and args.data.class_number == 100:
        ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
        net.load_state_dict(ckpt)
    else:
        net.load_state_dict(ckpt)

    net.cuda()

    # test_origin(net, test_dataloader)

    # Initialize group, clients and server.
    group = init_tta_state(args, net, train_dataloader, logger, corrupt_dict, corrupt_test_sets, origin_test_sets)

    # Training loop
    test_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]
    adapt_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]
    fed_adapt_monitor_list = [VariableMonitor() for _ in range(len(args.data.corruption))]

    if args.other.niid:
        loop_step = np.array([100, ] * args.client.client_num)
        length = args.client.client_num
        init_step = []
        for i in range(length):
            start, stop = (0, loop_step[i])
            init_step.append(random.randint(start, stop))

    test_times=0

    for level in args.data.level:
        for cidx in range(len(args.data.corruption)):
            # determine the corruption
            logger.logging('Starting Federated Test-Time Adaptation round: ')

            corupt_map = non_iid_continual(is_niid=args.other.niid, client_number=args.client.client_num,
                                           corupt_number=len(args.data.corruption))
            participated_clients = client_sampling(range(args.client.client_num), args.client.sample_rate)
            # Random sample participated clients in each communication round.
            # if cidx == 0 or not args.other.is_continue:
            if not args.other.is_continue:
                group = init_tta_state(args, net, train_dataloader, logger, corrupt_dict, corrupt_test_sets,
                                       origin_test_sets)

            global_eps = int(
                len(corrupt_test_sets[0][args.data.corruption[corupt_map[0][cidx]]][level]) / args.other.ttt_batch)
            for i in range(global_eps):
                if args.other.method == 'fed' or args.other.method == 'fedo' or args.other.method == 'fedbm' or args.other.method == 'bn':
                    global_mean = []
                    global_var = []
                if not args.other.online:
                    group = init_tta_state(args, net, train_dataloader, logger, corrupt_dict, corrupt_test_sets, origin_test_sets)

                for j in tqdm.tqdm(participated_clients):
                    if args.other.niid:
                        # if test_times % loop_step[j] == 0:
                        #     cor_type=random.randint(0,14)
                        #     corupt = args.data.corruption[corupt_map[cor_type][cidx]]
                        #
                        # else:
                            # corupt=old_cor_type
                        corupt = args.data.corruption[corupt_map[j][cidx]]
                        # while len(corrupt_test_sets[j][corupt][level].indexes) < args.other.ttt_batch:
                        #     corupt = (corupt + 1) % (args.data.corruption)
                    # Get the Coruption Type
                    else:
                        corupt = args.data.corruption[corupt_map[j][cidx]]

                    indexs = corrupt_test_sets[j][corupt][level].indexes[
                             0:  args.other.ttt_batch]
                    dataset = corrupt_test_sets[j][corupt][level].tot_data
                    corrupt_test_sets[j][corupt][level].indexes = corrupt_test_sets[j][corupt][level].indexes[args.other.ttt_batch : ]

                    inference_data = NaiveDataset(tot_data=dataset, indexes=indexs)

                    # Test Before Adaptation

                    test_monitor = group.clients[j].test_corupt(test_data=inference_data)
                    test_monitor_list[cidx].append(test_monitor)
                    logger.logging(
                        f'Client {j} Corupt {corupt}: Old Test Acc {test_monitor["test_acc"]}, Old Test Loss {test_monitor["test_loss"]}'
                    )
                    # Test Along with Adaptation
                    if args.other.method == 'tent':
                        adapt_monitor = group.clients[j].adapt_Tent(test_data=inference_data)
                        # adapt_monitor = group.clients[j].test_corupt(test_data=inference_data)
                    elif args.other.method == 'actmad':
                        adapt_monitor = group.clients[j].adapt_ActMAD(test_data=inference_data)
                    elif args.other.method == 'cotta':
                        adapt_monitor = group.clients[j].adapt_cotta(test_data=inference_data, ap=args.cotta.AP, mt=args.cotta.MT, rst=args.cotta.RST)
                    elif args.other.method == 't3a':
                        adapt_monitor = group.clients[j].adapt_T3A(test_data=inference_data)
                        logger.logging(
                            f'Client {j} Corupt {corupt} level {level}: New Test Acc {adapt_monitor["test_acc"]}, Model Predict Acc {adapt_monitor["test_acc_out"]}'
                        )
                    elif args.other.method == 'pl':
                        adapt_monitor = group.clients[j].adapt_PL(test_data=inference_data)
                    elif args.other.method == 'adanpc':
                        adapt_monitor = group.clients[j].adapt_AdaNPC(test_data=inference_data)
                    elif args.other.method == 'shot':
                        adapt_monitor = group.clients[j].adapt_SHOT(test_data=inference_data)
                    elif args.other.method == 'fed':
                        test_mean, test_var, adapt_monitor = group.clients[j].adapt_fed(test_data=inference_data)
                        global_mean.append(test_mean)
                        global_var.append(test_var)
                    elif args.other.method == 'sup':
                        adapt_monitor = group.clients[j].adapt_sup(test_data=inference_data)
                    elif args.other.method == 'bn':
                        test_mean, test_var, adapt_monitor = group.clients[j].adapt_BN(test_data=inference_data)
                        global_mean.append(test_mean)
                        global_var.append(test_var)

                    # if args.other.method != 'fed':
                    adapt_monitor_list[cidx].append(adapt_monitor)
                    if args.other.method != 't3a':
                        logger.logging(
                            f'Client {j} Corupt {corupt} level {level}: New Test Acc {adapt_monitor["test_acc"]}, New Test Loss {adapt_monitor["test_loss"]}'
                        )

                    test_times+=1
                    old_cor_type = corupt

                # Aggregate parameters in each client.
                if args.other.is_average:
                    logger.logging('-' * 10 + ' Average ' + '-' * 10)
                    if args.other.method == 'fed':
                        group.aggregate_bn(i, global_mean, global_var)
                    else:
                        trans_cost = group.aggregate(i)

                # if args.other.method == 'fed' or args.other.method == 'fedbm':
                for j in tqdm.tqdm(participated_clients):
                    if args.other.method == 'cotta':
                        adapt_monitor = group.clients[j].inference_cotta(ap=args.cotta.AP)
                    else:
                        adapt_monitor = group.clients[j].inference_fed()
                    fed_adapt_monitor_list[cidx].append(adapt_monitor)

                # if args.other.method == 'fed' or args.other.method == 'fedbm':
                logger.logging(
                    f'Coruption Type {corupt}, level {level}, Old Test Acc {adapt_monitor_list[cidx].variable_mean()["test_acc"]}\n,'
                    f'Fed Adapt Test Acc {fed_adapt_monitor_list[cidx].variable_mean()["test_acc"]}'
                )

    # Print & Save the outcome
    data_record = np.array([[0. for _ in range(len(args.data.corruption))] for _ in range(2)])
    for cidx in range(len(args.data.corruption)):
        corupt = args.data.corruption[cidx]
        # Origin
        mean_test_variables = test_monitor_list[cidx].variable_mean()
        data_record[0][cidx] = mean_test_variables["test_acc"]
        logger.logging(
            f'Corruption Type {corupt}, level {args.data.level}: Old Test Acc {mean_test_variables["test_acc"]}, Old Test Loss {mean_test_variables["test_loss"]}')

        if args.other.method == 'fed':
            mean_train_variables = fed_adapt_monitor_list[cidx].variable_mean()
        else:
            mean_train_variables = adapt_monitor_list[cidx].variable_mean()
        data_record[1][cidx] = mean_train_variables["test_acc"]

        if args.other.method == 't3a':
            logger.logging(
                f'Corruption Type {corupt}, level {args.data.level}: New Test Acc {mean_train_variables["test_acc"]}')
        else:
            logger.logging(
                f'Corruption Type {corupt}, level {args.data.level}: New Test Acc {mean_train_variables["test_acc"]}, New Test Loss {mean_train_variables["test_loss"]}')

    dfData = {
        '序号': ['Before', 'Adapt'],
    }

    for cidx in range(len(args.data.corruption)):
        corupt = args.data.corruption[cidx]
        dfData[corupt] = data_record[:, cidx]
    df = pd.DataFrame(dfData)
    df.to_excel(os.path.join(args.other.logging_path, 'outcome.xlsx'), index=False)


def generate_random_samples(client_num, corupt_num, corupt=True):
    arr = np.empty((client_num, corupt_num), dtype=int)
    if corupt:
        for i in range(client_num):
            arr[i] = np.random.permutation(corupt_num)
    else:
        for i in range(client_num):
            arr[i] = np.array([i for i in range(corupt_num)])
    return arr
