import copy

import matplotlib.pyplot as plt

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
from sklearn.manifold import TSNE

@torch.no_grad()
def sknopp(cZ, lamd=25, max_iters=100):
    N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
    probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

    r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
    c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

    inv_N_centroids = 1. / N_centroids
    inv_N_samples = 1. / N_samples

    err = 1e3
    for it in range(max_iters):
        r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
        c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
        if it % 10 == 0:
            err = torch.nansum(torch.abs(c / c_new - 1))
        c = c_new
        if (err < 1e-2):
            break

    # inplace calculations.
    probs *= c.squeeze()
    probs = probs.T # [N_samples, N_centroids]
    probs *= r.squeeze()

    return probs * N_samples # Soft assignments


def local_clustering(features, cluster_num):
    with torch.no_grad():
        Z = features
        centroids = Z[np.random.choice(Z.shape[0], cluster_num, replace=False)]
        local_iters = 5
        # clustering
        for it in range(local_iters):
            assigns = sknopp(Z @ centroids.T, max_iters=10)
            choice_cluster = torch.argmax(assigns, dim=1)
            for index in range(cluster_num):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                selected = torch.index_select(Z, 0, selected)
                if selected.shape[0] == 0:
                    selected = Z[torch.randint(len(Z), (1,))]
                centroids[index] = F.normalize(selected.mean(dim=0), dim=0)
            if it != 0:
                print(f"local cluster loss:{F.mse_loss(centroids, last_centroids)}")
            last_centroids = centroids.clone()

    feature_bank = torch.cat([features, centroids], dim=0)
    feature_bank = feature_bank.detach().cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    embeded_data = tsne.fit_transform(feature_bank)
    plt.scatter(embeded_data[:20, 0], embeded_data[:20, 1], s=5)
    plt.scatter(embeded_data[20:-10, 0], embeded_data[20:-10, 1], s=5)
    plt.scatter(embeded_data[-10:, 0], embeded_data[-10:, 1], s=10, marker='p')
    plt.title('Cluster')
    plt.show()
    return choice_cluster, centroids

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
        self.history_feature = [[] for _ in range(args.client.client_num)]
        self.history_weight = [[] for _ in range(args.client.client_num)]
        self.indicator = torch.tensor([])
        self.time_slide = 10


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

    def topk_softmax(self, weight, k, num):
        topk_values, topk_indices = torch.topk(weight, num)
        topk_values = weight[topk_indices]
        softmax_values = torch.softmax(topk_values, dim=0)
        return softmax_values, topk_indices

    def st_agg_grad(self, time_att, space_att):
        client_num = self.args.client.client_num
        time_att = torch.mean(time_att, dim=1)
        weight_list = []
        for cidx in range(client_num):
            w_time = copy.deepcopy(self.clients[cidx].model.state_dict())
            T_all = time_att.shape[2]
            for k in w_time.keys():
                for tidx in range(T_all):
                    if tidx == 0:
                        w_time[k] = time_att[cidx, -1, tidx] * self.history_weight[cidx][-T_all + tidx][k].cuda()
                    else:
                        w_time[k] += time_att[cidx, -1, tidx] * self.history_weight[cidx][-T_all + tidx][k].cuda()
            weight_list.append(w_time)

        for cidx in range(client_num):
            w_space = copy.deepcopy(self.clients[cidx].model.state_dict())
            S_all = space_att.shape[0]
            for k in w_space.keys():
                for sidx in range(S_all):
                    if sidx == 0:
                        w_space[k] = space_att[cidx, sidx] * weight_list[sidx][k]
                    else:
                        w_space[k] += space_att[cidx, sidx] * weight_list[sidx][k]

            self.clients[cidx].model.load_state_dict(w_space)


    def aggregate_grad(self, train_round, feature_indicator):
        client_num = self.args.client.client_num
        for cidx in range(client_num):
            self.history_weight[cidx].append(self.clients[cidx].model.state_dict())

        if self.args.group.aggregation_method == 'st':
            time_att, space_att = self.ST_attention(feature_indicator)
            self.st_agg_grad(time_att, space_att)

        elif self.args.group.aggregation_method == 'avg':
            total_samples = float(sum([client.sample_num for client in self.clients]))
            w_avg = copy.deepcopy(self.clients[0].model.state_dict())
            for key in w_avg.keys():
                for cidx in range(client_num):
                    if cidx == 0:
                        continue
                    else:
                        w_avg[key] += self.clients[cidx].model.state_dict()[key]
                w_avg[key] = torch.div(w_avg[key], client_num)
            for client in self.clients:
                client.model.load_state_dict(w_avg)

        elif self.args.group.aggregation_method == 'sim':
            time_att, space_att = self.ST_similarity(feature_indicator)
            self.st_agg_grad(time_att, space_att)

        elif self.args.group.aggregation_method == 'wotime':
            time_att, space_att = self.ST_attention(feature_indicator, wotime=True)
            self.st_agg_grad(time_att, space_att)

    def st_agg_bn(self, time_att, space_att, global_mean):
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        for chosen_layer in range(n_chosen_layer):
            if self.history_feature[0].shape[1] < self.time_slide:
                feature_input = self.history_feature[chosen_layer][:, 1:, :]
            else:
                T_all = self.history_feature[chosen_layer].shape[1]
                feature_input = self.history_feature[chosen_layer][:, T_all - self.time_slide:, :]
            N, T, D = feature_input.shape
            heads = 1
            out = torch.matmul(time_att, feature_input.view(N, T, heads, D).permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous().view(N, T, heads * D)
            out = torch.matmul(space_att, out.view(N, T, heads, D).permute(1, 2, 0, 3)).permute(1, 2, 0, 3).contiguous().view(N, T, heads * D)

            half = out.shape[2] // 2
            for cidx in range(client_num):
                sum_mean[cidx][chosen_layer] = out[cidx][T - 1][:half]
                sum_var[cidx][chosen_layer] = out[cidx][T - 1][half:]
        return sum_mean, sum_var

    def aggregate_bn(self, train_round, global_mean, feature_indicator):
        # Store feature mean and variance
        n_chosen_layer = len(global_mean[0])
        client_num = self.args.client.client_num
        for chosen_layer in range(n_chosen_layer):
            feature_t = []
            for cidx in range(client_num):
                feature_t.append(global_mean[cidx][chosen_layer])
            feature_t = torch.stack(feature_t, dim=0)
            self.history_feature[chosen_layer] = torch.cat([self.history_feature[chosen_layer], feature_t.unsqueeze(1)], dim=1)

        # calculate aggregation rate & aggregate model weight
        sum_mean = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        sum_var = [[[] for _ in range(n_chosen_layer)] for _ in range(client_num)]
        if self.args.group.aggregation_method == 'st':
            time_att, space_att = self.ST_attention(feature_indicator)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean)

        elif self.args.group.aggregation_method == 'avg':
            total_samples = float(sum([client.sample_num for client in self.clients]))
            for chosen_layer in range(len(global_mean[0])):
                half = len(global_mean[0][chosen_layer]) // 2
                for idx in range(len(global_mean)):
                    sum_mean[chosen_layer] += global_mean[idx][chosen_layer][:half] * self.clients[idx].sample_num
                    sum_var[chosen_layer] += global_mean[idx][chosen_layer][half:] * self.clients[idx].sample_num

                sum_mean[chosen_layer] /= total_samples
                sum_var[chosen_layer] /= total_samples

        elif self.args.group.aggregation_method == 'sim':
            time_att, space_att = self.ST_similarity(feature_indicator)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean)
        
        elif self.args.group.aggregation_method == 'wotime':
            time_att, space_att = self.ST_attention(feature_indicator, wotime=True)
            sum_mean, sum_var = self.st_agg_bn(time_att, space_att, global_mean)

        for cidx in range(client_num):
            self.clients[cidx].update_bnstatistics(sum_mean[cidx], sum_var[cidx])

    def ST_attention(self, feature_indicator, wotime=False):
        '''
        :param feature_indicator: global_mean[ cidx ][ chosen_layer ][ D(mean) ]
        :return: weight1, weight2
        '''

        feature_indicator = torch.stack(feature_indicator, dim=0)
        if self.indicator.shape[0] == 0:
            self.indicator = feature_indicator.unsqueeze(1)
        else:
            self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)

        # Get Aggregate Weights with Trainable Modules
        self.time_slide = 10
        if self.indicator.shape[1] < self.time_slide:
            feature_input = self.indicator[:, :, :]
        else:
            feature_input = self.indicator[:, self.indicator.shape[1]-self.time_slide:, :]
        ST_model = ST_block(dim=feature_input.shape[2])
        ST_model.cuda()
        opt = torch.optim.Adam(ST_model.parameters(), lr=1e-3)
        loss_min = 1000000
        for epoch in range(100):
            print('Epoch {}'.format(epoch))
            ST_model.train()
            logits, mask_logits, aug_logits, t_sim, s_sim = ST_model(feature_input, wotime=wotime)
            loss_reg = F.mse_loss(feature_input, logits)
            loss_consist = F.mse_loss(logits, mask_logits)
            loss_robust = F.mse_loss(logits, aug_logits)

            loss = (loss_reg + loss_robust) * 100

            if loss.item() < loss_min:
                time_att = t_sim
                space_att = s_sim
                loss_min = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            print('loss = {:.4f} + {:.4f} + {:.4f} = {:.4f}(min {:.4f})'.format(loss_reg.item(), loss_consist.item(),
                                                                                loss_robust.item(), loss.item(),
                                                                                loss_min))

        return time_att, space_att

    def ST_similarity(self, feature_indicator):
        feature_indicator = torch.stack(feature_indicator, dim=0)
        self.indicator = torch.cat([self.indicator, feature_indicator.unsqueeze(1)], dim=1)

        self.time_slide = 10
        if self.indicator.shape[1] < self.time_slide:
            feature_input = self.indicator[:, 1:, :]
        else:
            feature_input = self.indicator[:, self.indicator.shape[1] - self.time_slide:, :]
        ST_model = ST_block(dim=feature_input.shape[2])
        ST_model.cuda()
        logits, mask_logits, aug_logits, t_sim, s_sim = ST_model(feature_input)

        time_att = t_sim
        space_att = s_sim

        return time_att, space_att


