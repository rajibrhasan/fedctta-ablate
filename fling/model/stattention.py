import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from sklearn.cluster import KMeans

class TempoAttention(nn.Module):
    def __init__(self, dim, heads, dropout=.1):
        super(TempoAttention, self).__init__()
        self.fc_q = nn.Linear(dim, heads * dim)
        self.fc_k = nn.Linear(dim, heads * dim)
        self.fc_v = nn.Linear(dim, heads * dim).requires_grad_(False)
        self.dropout = nn.Dropout(dropout)

        self.dim = dim
        self.heads = heads

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.eye_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, attention_weights=None):
        N, T, D = x.shape

        q = self.fc_q(x).view(N, T, self.heads, D).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(x).view(N, T, self.heads, D).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(x).view(N, T, self.heads, D).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        self.att = torch.matmul(q, k) / np.sqrt(D)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            self.att = self.att * attention_weights
        if attention_mask is not None:
            self.att = self.att.masked_fill(attention_mask, -np.inf)
        self.att = torch.softmax(self.att, -1)
        att = self.dropout(self.att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(N, T, self.heads * D)  # (b_s, nq, h*d_v)

        return out

def graphStructual(x):
    avg_feature = torch.mean(x, dim=1)
    adj_matrix = torch.matmul(avg_feature, avg_feature.t())
    denominator = torch.outer(torch.norm(avg_feature, dim=1), torch.norm(avg_feature, dim=1))
    similarity_matrix = adj_matrix / denominator
    similarity_mask = (similarity_matrix > 0.9).float()
    return similarity_mask

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

class SpatialAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0., qkv_bias=True):
        super(SpatialAttention, self).__init__()
        self.geo_q = nn.Linear(dim, heads * dim, bias=qkv_bias)
        self.geo_k = nn.Linear(dim, heads * dim, bias=qkv_bias)
        self.geo_v = nn.Linear(dim, heads * dim, bias=qkv_bias).requires_grad_(False)
        self.dropout = nn.Dropout(dropout)

        self.dim = dim
        self.heads = heads

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.eye_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        N, T, D = x.shape

        q = self.geo_q(x).view(N, T, self.heads, D).permute(1, 2, 0, 3)  # (T, head, N, D)
        k = self.geo_k(x).view(N, T, self.heads, D).permute(1, 2, 3, 0)  # (T, head, D, N)
        v = self.geo_v(x).view(N, T, self.heads, D).permute(1, 2, 0, 3)   # (T, head, N, D)

        att = torch.matmul(q, k) / np.sqrt(D)  # (T, head, D, D)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(1, 2, 0, 3).contiguous().view(N, T, self.heads * D)  # (b_s, nq, h*d_v)

        return out

class STProject(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, drop=0.):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.proj_drop = nn.Dropout(drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.eye_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        out = self.proj(x)
        out = self.proj_drop(out)
        return out

class ClusterLoss(nn.Module):
    def __init__(self, num_clusters, distance_threshold=1.0):
        super(ClusterLoss, self).__init__()
        self.num_clusters = num_clusters
        self.distance_threshold = distance_threshold

    def forward(self, features, cluster_assignments, cluster_centers):
        # 计算每个节点与其所属簇中心的欧氏距离
        dist_to_assigned_center = torch.norm(features - cluster_centers[cluster_assignments], dim=1)

        # 计算每个节点与其他簇中心的欧氏距离
        dist_to_other_centers = torch.cat([torch.norm(features - center, dim=1, keepdim=True) for center in cluster_centers], dim=1)
        dist_to_other_centers.scatter_(1, cluster_assignments.view(-1, 1), float('inf'))  # 将自身簇的距离设为无穷大

        # 计算损失
        loss = torch.sum(torch.relu(self.distance_threshold - dist_to_assigned_center)) + torch.sum(torch.relu(dist_to_other_centers - self.distance_threshold))

        return loss / features.size(0)  # 归一化损失

class ST_block(nn.Module):
    def __init__(self, dim, Theads=1, Sheads=1, proj_bias=False, proj_dropout=.0):
        super().__init__()
        self.dim = dim

        self.TA1 = TempoAttention(dim=dim, heads=Theads)
        self.SA1 = SpatialAttention(dim=dim, heads=Sheads)
        self.TA2 = TempoAttention(dim=dim, heads=Theads)
        self.SA2 = SpatialAttention(dim=dim, heads=Sheads)
        # self.Proj = STProject(input_dim=dim, output_dim=dim, bias=proj_bias, drop=proj_dropout)

    def forward(self, x):
        TA_x = self.TA1(x)
        SA_x = self.SA1(TA_x)
        TA_x = self.TA2(SA_x)
        SA_x = self.SA2(TA_x)
        # ST_x = self.Proj(SA_x)
        return SA_x



if __name__ == '__main__':
    N = 50
    T = 20
    D = 64
    input = torch.randn(50, 20, 64)
    TA = TempoAttention(dim=64, heads=1)
    TA_output = TA(input)

    SA = SpatialAttention(dim=64, heads=1)
    SA_output = SA(TA_output)

    STP = STProject(input_dim=64, output_dim=64, bias=False)
    ST_output = STP(SA_output)

    num_clusters = 3
    data = ST_output.detach().clone()
    data = data.view(N * T, D).numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(data)
    cluster_labels = torch.from_numpy(cluster_labels).view(N, T)
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

    loss_reg = F.mse_loss(input, ST_output)

    ST_output_flat = ST_output.view(N * T, D)
    cluster_centers_flat = cluster_centers.view(num_clusters, D)
    cluster_center4feature = cluster_centers_flat.index_select(0, cluster_labels.view(-1))
    loss_global = torch.mean(torch.norm(ST_output_flat - cluster_center4feature, dim=1))

    loss = loss_reg + loss_global












