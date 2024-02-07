import time
import copy
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup
import torch.nn as nn

import numpy as np
import cvxpy as cp

@GROUP_REGISTRY.register('ldawa_group')
class LDAWAServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedCAC.
    """

    def __init__(self, args: dict, logger: Logger):
        super(LDAWAServerGroup, self).__init__(args, logger)
        # To be consistent with the existing pipeline interface. group maintains an epoch counter itself.
        self.epoch = -1
        self.client_num = self.args.client.client_num
        self.graph_matrix = torch.ones(self.client_num, self.client_num) / (self.client_num - 1)  # Collaboration Graph
        self.graph_matrix[range(self.client_num), range(self.client_num)] = 0
        self.dw = []
        self.glob_model = None

    def weight_flatten(self, layer):

        return layer.reshape(-1)

    def aggregate_grad(self,  train_round, feature_indicator):
        if train_round == 0:
            total_samples = float(sum([client.sample_num for client in self.clients]))
            w_avg = copy.deepcopy(self.clients[0].model.state_dict())
            for key in w_avg.keys():
                for cidx in range(self.client_num):
                    if cidx == 0:
                        continue
                    else:
                        w_avg[key] += self.clients[cidx].model.state_dict()[key]
                w_avg[key] = torch.div(w_avg[key], self.client_num)
            for client in self.clients:
                client.model.load_state_dict(w_avg)
            self.glob_model = copy.deepcopy(self.clients[0].model)
        else:
            cosin = nn.CosineSimilarity(dim=0)
            glob_state_dict = self.glob_model.state_dict()
            clients_state_dict = [self.clients[i].model.state_dict() for i in range(self.client_num)]
            layer_name_list = glob_state_dict.keys()
            for cidx in range(self.client_num):
                for layer_name in layer_name_list:
                    x = clients_state_dict[cidx][layer_name]
                    y = glob_state_dict[layer_name]
                    if len(x.shape) and len(y.shape) > 0:
                        flatten_x = torch.unsqueeze(self.weight_flatten(x), dim=1)
                        flatten_y = torch.unsqueeze(self.weight_flatten(y), dim=1)
                        v = cosin(flatten_x, flatten_y)
                        clients_state_dict[cidx][layer_name] = v * x
                    else:
                        clients_state_dict[cidx][layer_name] = x
            w_avg = copy.deepcopy(clients_state_dict[0])
            for key in w_avg.keys():
                for cidx in range(self.client_num):
                    if cidx == 0:
                        continue
                    else:
                        w_avg[key] += clients_state_dict[cidx][key]
                w_avg[key] = torch.div(w_avg[key], self.client_num)
            for client in self.clients:
                client.model.load_state_dict(w_avg)
            self.glob_model.load_state_dict(w_avg)

    def aggregate_bn(self, train_round, global_mean, feature_indicator):
        self.graph_matrix = self.update_graph_matrix_neighbor_bn(global_mean, similarity_matric='all')

        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        out = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]

        for cidx in range(self.client_num):
            for chosen_layer in range(n_chosen_layer):
                for sidx in range(self.client_num):
                    if sidx == 0:
                        out[cidx][chosen_layer] = global_mean[sidx][chosen_layer] * self.graph_matrix[cidx][sidx]
                    else:
                        out[cidx][chosen_layer] += global_mean[sidx][chosen_layer] * self.graph_matrix[cidx][sidx]

                half = out[cidx][chosen_layer].shape[0] // 2
                for cidx in range(client_num):
                    sum_mean[cidx][chosen_layer] = out[cidx][chosen_layer][:half]
                    sum_var[cidx][chosen_layer] = out[cidx][chosen_layer][half:]
            self.clients[cidx].update_bnstatistics(sum_mean[cidx], sum_var[cidx])

