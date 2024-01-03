import time
import copy
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup

import numpy as np
import cvxpy as cp

@GROUP_REGISTRY.register('fedamp_group')
class FedAMPServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedCAC.
    """

    def __init__(self, args: dict, logger: Logger):
        super(FedAMPServerGroup, self).__init__(args, logger)
        # To be consistent with the existing pipeline interface. group maintains an epoch counter itself.
        self.epoch = -1
        self.client_num = self.args.client.client_num
        self.graph_matrix = torch.ones(self.client_num, self.client_num) / (self.client_num - 1)  # Collaboration Graph
        self.graph_matrix[range(self.client_num), range(self.client_num)] = 0
        self.dw = []

    def weight_flatten(self, model):
        params = []
        for k in model:
            if 'fc' in k:
                params.append(model[k].reshape(-1))
        params = torch.cat(params)
        return params

    def weight_flatten_all(self, model):
        params = []
        for k in model:
            params.append(model[k].reshape(-1))
        params = torch.cat(params)
        return params

    def cal_model_cosine_difference(self, ckpt):
        model_similarity_matrix = torch.zeros((self.client_num, self.client_num))
        for i in range(self.client_num):
            self.dw.append({key: torch.zeros_like(value) for key, value in self.clients[0].model.named_parameters()})
        for cidx in range(self.client_num):
            model_i = self.clients[cidx].model.state_dict()
            for key in model_i.keys():
                self.dw[cidx][key] = model_i[key] - ckpt[key]

        for i in range(self.client_num):
            for j in range(i, self.client_num):
                if i == j:
                    similarity = 0
                else:
                    similarity = torch.norm((self.weight_flatten_all(self.dw[i]).unsqueeze(0)-
                                             self.weight_flatten_all(self.dw[j]).unsqueeze(0)), p=2)
                model_similarity_matrix[i, j] = similarity
                model_similarity_matrix[j, i] = similarity
        return model_similarity_matrix

    # def update_graph_matrix_neighbor(self, ckpt):
    #     model_difference_matrix = self.cal_model_cosine_difference(ckpt)
    #     graph_matrix = self.calculate_graph_matrix(model_difference_matrix)
    #     print(f'Model difference: {model_difference_matrix[0]}')
    #     print(f'Graph matrix: {graph_matrix}')
    #     return graph_matrix

    def calculate_graph_matrix(self, model_difference_matrix):
        graph_matrix = torch.zeros((model_difference_matrix.shape[0], model_difference_matrix.shape[0]))
        self_weight = 0.3
        for i in range(model_difference_matrix.shape[0]):
            weight = torch.exp(-model_difference_matrix[i])
            weight[i] = 0
            weight = (1 - self_weight) * weight / weight.sum()
            weight[i] = self_weight
            graph_matrix[i] = weight

        return graph_matrix

    def update_graph_matrix_neighbor(self, feature_indicator):
        model_similarity_matrix = torch.zeros((self.client_num, self.client_num))
        for i in range(self.client_num):
            for j in range(i, self.client_num):
                if i == j:
                    similarity = 0
                else:
                    similarity = torch.norm((feature_indicator[i].unsqueeze(0) -
                                             feature_indicator[j].unsqueeze(0)), p=2)
                model_similarity_matrix[i, j] = similarity
                model_similarity_matrix[j, i] = similarity

        graph_matrix = self.calculate_graph_matrix(model_similarity_matrix)
        print(f'Model difference: {model_similarity_matrix[0]}')
        print(f'Graph matrix: {graph_matrix}')
        return graph_matrix

    def aggregate_grad(self,  train_round, feature_indicator):
        # self.graph_matrix = self.update_graph_matrix_neighbor(self.server.glob_dict)
        self.graph_matrix = self.update_graph_matrix_neighbor(feature_indicator)
        tmp_client_state_dict = {}
        for cidx in range(self.client_num):
            tmp_client_state_dict[cidx] = copy.deepcopy(self.server.glob_dict)
            for key in tmp_client_state_dict[cidx]:
                tmp_client_state_dict[cidx][key] = torch.zeros_like(tmp_client_state_dict[cidx][key])

        for cidx in range(self.client_num):
            tmp_client_state = tmp_client_state_dict[cidx]
            aggregation_weight_vector = self.graph_matrix[cidx]

            for cidx1 in range(self.client_num):
                net_para = self.clients[cidx1].model.state_dict()
                for key in tmp_client_state:
                    if 'num_batches_tracked' not in key:
                        tmp_client_state[key] += net_para[key] * aggregation_weight_vector[cidx1]

        for cidx in range(self.client_num):
            self.clients[cidx].model.load_state_dict(tmp_client_state_dict[cidx])

    def update_graph_matrix_neighbor_bn(self, global_mean, similarity_matric, lamba=0.8):
        dw = []
        for cidx in range(len(global_mean)):
            for n_chosen_layer in range(len(global_mean[0])):
                if n_chosen_layer == 0:
                    tmp_bn = global_mean[cidx][n_chosen_layer]
                else:
                    tmp_bn = torch.cat([tmp_bn, global_mean[cidx][n_chosen_layer]], dim=0)
            dw.append(tmp_bn)

        model_difference_matrix = torch.zeros([self.client_num, self.client_num])
        for i in range(self.client_num):
            for j in range(i, self.client_num):
                diff = - torch.nn.functional.cosine_similarity(dw[i].unsqueeze(0), dw[j].unsqueeze(0))
                # if diff < -0.9:
                #     diff = -1.0
                model_difference_matrix[i, j] = diff
                model_difference_matrix[j, i] = diff

        total_data_points = sum([self.clients[k].sample_num for k in range(self.client_num)])
        fed_avg_freqs = {k: self.clients[k].sample_num / total_data_points for k in range(self.client_num)}

        n = model_difference_matrix.shape[0]
        p = np.array(list(fed_avg_freqs.values()))
        P = lamba * np.identity(n)
        P = cp.atoms.affine.wraps.psd_wrap(P)
        G = - np.identity(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.ones(1)
        for i in range(model_difference_matrix.shape[0]):
            model_difference_vector = model_difference_matrix[i]
            d = model_difference_vector.numpy()
            q = d - 2 * lamba * p
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                              [G @ x <= h,
                               A @ x == b]
                              )
            prob.solve()

            self.graph_matrix[i, :] = torch.Tensor(x.value)

        return self.graph_matrix

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

