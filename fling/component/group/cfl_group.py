import time
import copy
import torch

from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import Logger
from fling.component.group import ParameterServerGroup

import numpy as np
import cvxpy as cp
from sklearn.cluster import AgglomerativeClustering

@GROUP_REGISTRY.register('cfl_group')
class CFLServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Implementation of the group in FedCAC.
    """

    def __init__(self, args: dict, logger: Logger):
        super(CFLServerGroup, self).__init__(args, logger)
        # To be consistent with the existing pipeline interface. group maintains an epoch counter itself.
        self.epoch = -1
        self.client_num = self.args.client.client_num
        self.dw = []
        self.n_parties = 4

    def initialize(self) -> None:
        r"""
        Overview:
            In this function, several things will be done:
            1) Set ``fed_key`` in each client is determined, determine which parameters shoul be included for federated
        learning.
            2) ``glob_dict`` in the server is determined, which is exactly a state dict with all keys in ``fed_keys``.
            3) Each client local model will be updated by ``glob_dict``.
        Returns:
            - None
        """
        # Step 1.
        if self.args.group.aggregation_parameters.name == 'all':
            fed_keys = self.clients[0].model.state_dict().keys()
        elif self.args.group.aggregation_parameters.name == 'contain':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'except':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for kw in keywords:
                for k in self.clients[0].model.state_dict():
                    if kw in k:
                        fed_keys.append(k)
            fed_keys = list(set(self.clients[0].model.state_dict().keys()) - set(fed_keys))
        elif self.args.group.aggregation_parameters.name == 'include':
            keywords = self.args.group.aggregation_parameters.keywords
            fed_keys = []
            for name, param in self.clients[0].model.named_parameters():
                if keywords in name:
                    fed_keys.append(name)
            fed_keys = list(set(fed_keys))
        else:
            raise ValueError(f'Unrecognized aggregation_parameters.name: {self.args.group.aggregation_parameters.name}')

        # Step 2.
        self.logger.logging(f'Weights for federated training: {fed_keys}')
        glob_dict = {k: self.clients[0].model.state_dict()[k] for k in fed_keys}
        self.server.glob_dict = glob_dict
        self.set_fed_keys()

        # Step 3.
        if not self.args.other.resume:
            self.sync()

        # Step 4. CFL
        self.weight_past = [self.clients[i].model.state_dict() for i in range(self.args.client.client_num)]
        self.cluster_indices = [np.arange(self.args.client.client_num).astype("int")]
        self.client_clusters = [[self.clients[i].model for i in idcs] for idcs in self.cluster_indices]

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')

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

    def cal_model_cosine_difference(self, model_weight_past):
        model_similarity_matrix = torch.zeros((self.client_num, self.client_num))
        for i in range(self.client_num):
            self.dw.append({key: torch.zeros_like(value) for key, value in self.clients[0].model.named_parameters()})
        for cidx in range(self.client_num):
            model_i = self.clients[cidx].model.state_dict()
            for key in model_i.keys():
                self.dw[cidx][key] = model_i[key] - model_weight_past[cidx][key]

        for i in range(self.client_num):
            for j in range(i, self.client_num):
                simi = torch.nn.functional.cosine_similarity(
                    self.weight_flatten_all(self.dw[i]).unsqueeze(0),
                    self.weight_flatten_all(self.dw[j]).unsqueeze(0))

                model_similarity_matrix[i, j] = simi
                model_similarity_matrix[j, i] = simi
        torch.cuda.empty_cache()
        return model_similarity_matrix

    def compute_max_update_norm(self, cluster):
        norm_list = []
        for client_dw in cluster:
            weight = self.weight_flatten_all(client_dw)
            norm_list.append(torch.norm(weight).item())
        norm_list = np.array(norm_list)
        return np.max(norm_list)
        # return np.max([torch.norm(self.weight_flatten(client_dw)).item() for client_dw in cluster])

    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([self.weight_flatten(client_dw) for client_dw in cluster]), dim=0)).item()

    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten()
        c2 = np.argwhere(clustering.labels_ == 1).flatten()
        return c1, c2

    def reduce_add_average(self, targets, sources):
        for target in targets:
            for k, v in target.named_parameters():
                tmp = torch.mean(torch.stack([source[k].data for source in sources]), dim=0).clone()
                # v.data += tmp
                v.data = tmp

    def compute_max_simi(self, similarity, idc):
        min_simi = 1000
        for i in idc:
            for j in idc:
                if i != j and similarity[i, j] < min_simi:
                    min_simi = similarity[i, j]
        return min_simi

    def aggregate_grad(self,  train_round, feature_indicator, eps2=0.01):

        similarity = self.cal_model_cosine_difference(self.weight_past)
        print(similarity)

        cluster_indices_new = []
        for idc in self.cluster_indices:
            max_norm = self.compute_max_update_norm([self.dw[i] for i in idc])
            # mean_norm = self.compute_mean_update_norm([self.dw[i] for i in idc])
            alpha = self.compute_max_simi(similarity, idc)
            print(alpha, max_norm, eps2)
            if max_norm > eps2 and len(idc) > 2 and alpha < 0.97:
                c1, c2 = self.cluster_clients(similarity[idc][:, idc])
                print("new split", idc, c1, c2)
                cluster_indices_new += [idc[c1], idc[c2]]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[self.clients[i].model for i in idcs] for idcs in cluster_indices]
        # gradient_clusters = [[self.dw[i] for i in idcs] for idcs in self.cluster_indices]
        gradient_clusters = [[copy.deepcopy(self.clients[i].model.state_dict()) for i in idcs] for idcs in cluster_indices]
        for i in range(len(cluster_indices)):
            self.reduce_add_average(client_clusters[i], gradient_clusters[i])
        print("cluster", cluster_indices)

        self.weight_past = [self.clients[i].model.state_dict() for i in range(self.client_num)]

    def aggregate_bn(self, train_round, global_mean, feature_indicator, eps2=0.01):
        similarity = self.cal_model_cosine_difference(self.weight_past)
        print(similarity)

        cluster_indices_new = []
        for idc in self.cluster_indices:
            max_norm = self.compute_max_update_norm([self.dw[i] for i in idc])
            # mean_norm = self.compute_mean_update_norm([self.dw[i] for i in idc])
            alpha = self.compute_max_simi(similarity, idc)
            print(alpha, max_norm, eps2)
            if max_norm > eps2 and len(idc) > 2 and alpha < 0.97:
                c1, c2 = self.cluster_clients(similarity[idc][:, idc])
                print("new split", idc, c1, c2)
                cluster_indices_new += [idc[c1], idc[c2]]
            else:
                cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        client_clusters = [[self.clients[i].model for i in idcs] for idcs in cluster_indices]
        # gradient_clusters = [[self.dw[i] for i in idcs] for idcs in self.cluster_indices]
        gradient_clusters = [[copy.deepcopy(self.clients[i].model.state_dict()) for i in idcs] for idcs in
                             cluster_indices]
        for i in range(len(cluster_indices)):
            self.reduce_add_average(client_clusters[i], gradient_clusters[i])
        print("cluster", cluster_indices)
        # Store feature mean and variance
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        # calculate aggregation rate & aggregate model weight
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]



