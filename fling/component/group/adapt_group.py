from fling.utils import get_params_number
from fling.utils.compress_utils import *
from fling.utils.registry_utils import GROUP_REGISTRY
from fling.utils import VariableMonitor
from fling.component.client import ClientTemplate
import torch
from functools import reduce
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from fling.model.stattention import ST_block
from fling.component.group import ParameterServerGroup

@GROUP_REGISTRY.register('adapt_group')
class TTAServerGroup(ParameterServerGroup):
    r"""
    Overview:
        Base implementation of the group in federated learning.
    """

    def __init__(self, args: dict, logger: VariableMonitor):
        r"""
        Overview:
            Implementation of the group in FedTTA
        """
        super(TTAServerGroup, self).__init__(args, logger)
        self.history_mean = [[] for _ in range(args.client.client_num)]
        self.history_var = [[] for _ in range(args.client.client_num)]
        self.history_feature = [[] for _ in range(args.client.client_num)]
        self.bank_mean = []
        self.bank_var = []
        self.layer_bank = []
        self.history = []
        self.past_mean = []
        self.past_var = []


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

        # Logging model information.
        self.logger.logging(str(self.clients[0].model))
        self.logger.logging('All clients initialized.')
        self.logger.logging(
            'Parameter number in each model: {:.2f}M'.format(get_params_number(self.clients[0].model) / 1e6)
        )

    def append(self, client: ClientTemplate) -> None:
        r"""
        Overview:
            Append a client into the group.
        Arguments:
            - client: client to be added.
        Returns:
            - None
        """
        self.clients.append(client)


    def aggregate_fedbm(self):
        total_samples = sum([client.sample_num for client in self.clients])
        # Weighted-averaging.
        self.server.glob_dict = {
            k: reduce(
                lambda x, y: x + y,
                [client.model.state_dict()[k] for client in self.clients]
            )
            for k in self.clients[0].fed_keys
        }
        self.sync()

    def aggregate(self, train_round: int) -> int:
        r"""
        Overview:
            Aggregate all client models.
        Arguments:
            - train_round: current global epochs.
        Returns:
            - trans_cost: uplink communication cost.
        """
        if self.args.group.aggregation_method == 'avg':
            trans_cost = fed_avg(self.clients, self.server)
            self.sync()
        else:
            print('Unrecognized compression method: ' + self.args.group.aggregation_method)
            assert False
        return trans_cost

    def init_bnstatistics(self, mean, var):
        # for cidx in range(self.args.client.client_num):
        #     for chosen_layer in range(len(mean)):
        #         self.history_mean[cidx].append(mean[chosen_layer].view(1, -1))
        #         self.history_var[cidx].append(var[chosen_layer].view(1, -1))
        feature_clean = []
        for chosen_layer in range(len(mean)):
            feature_clean.append(torch.cat([mean[chosen_layer], var[chosen_layer]], dim=0))
        feature_clean = torch.stack(feature_clean, dim=0)
        for cidx in range(self.args.client.client_num):
            self.history_feature[cidx] = feature_clean.view(1, -1)
        self.history_feature = torch.stack(self.history_feature, dim=0)


    def topk_softmax(self, weight, k, num):
        topk_values, topk_indices = torch.topk(weight, num)
        topk_values = weight[topk_indices]
        softmax_values = torch.softmax(topk_values, dim=0)
        return softmax_values, topk_indices

    def aggregate_bn(self, train_round: int, global_mean, global_var):
        self.ST_attention(global_mean, global_var)
        return

    def ST_attention(self, global_mean, global_var):
        feature_t = []
        # feature_t_mean, feature_t_var = [], []
        for cidx in range(self.args.client.client_num):
            for chosen_layer in range(len(global_mean[0])):
                feature_t.append(torch.cat([global_mean[cidx][chosen_layer], global_var[cidx][chosen_layer]], dim=0))
                # feature_t_mean.append(global_mean[cidx][chosen_layer])
                # feature_t_var.append(global_var[cidx][chosen_layer])
        feature_t = torch.stack(feature_t, dim=0)
        # feature_t_mean = torch.stack(feature_t_mean, dim=0)
        # feature_t_var = torch.stack(feature_t_var, dim=0)
        self.history_feature = torch.cat((self.history_feature, feature_t.unsqueeze(1)), dim=1)
        # self.history_feature_mean = torch.cat((self.history_feature_mean, feature_t_mean.unsqueeze(1)), dim=1)
        # self.history_feature_var = torch.cat((self.history_feature_var, feature_t_var.unsqueeze(1)), dim=1)
        if self.history_feature.shape[1] < 20:
            feature_input = self.history_feature
        else:
            feature_input = self.history_feature[:, self.history_feature.shape[1]-20:, :]
        ST_model = ST_block(dim=self.history_feature.shape[2])
        ST_model.cuda()
        opt = torch.optim.Adam(ST_model.parameters(), lr=1e-5)
        loss_min = 1000000
        for epoch in range(20):
            print('Epoch {}'.format(epoch))
            ST_model.train()
            logits = ST_model(feature_input)

            N, T, D = logits.shape
            num_clusters = 3
            data = logits.detach().cpu().clone()
            data = data.view(N * T, D).numpy()
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(data)
            cluster_labels = torch.from_numpy(cluster_labels).view(N, T)
            cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

            loss_reg = F.mse_loss(feature_input, logits)

            ST_output_flat = logits.view(N * T, D)
            cluster_centers_flat = cluster_centers.view(num_clusters, D)
            cluster_center4feature = cluster_centers_flat.index_select(0, cluster_labels.view(-1))
            cluster_center4feature = cluster_center4feature.cuda()
            loss_global = torch.mean(torch.norm(ST_output_flat - cluster_center4feature, dim=1))

            loss = (loss_global + loss_reg)*100

            if loss.item() < loss_min:
                logits_best = logits
                loss_min = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('loss = {:.4f} + {:.4f} = {:.4f}(min {:.4f})'.format(loss_global.item(), loss_reg.item(), loss.item(), loss_min))

            half_dim = D // 2
            for cidx in range(self.args.client.client_num):
                self.clients[cidx].update_bnstatistics(logits_best[cidx, T-1, :half_dim].detach(), logits_best[cidx, T-1, half_dim:].detach())
                aggregate_monitor = self.clients[cidx].inference_fed()
                print(f'{cidx}, epoch {epoch}, loss {aggregate_monitor["test_loss"]}, acc {aggregate_monitor["test_acc"]}')


