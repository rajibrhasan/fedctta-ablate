import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Iterable

from fling.utils.registry_utils import CLIENT_REGISTRY
from .client_template import ClientTemplate
from fling.model import get_model
from fling.utils.utils import VariableMonitor

import torch.nn.functional as F

@CLIENT_REGISTRY.register('fedshot_client')
class FedSHOTClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedSHOTClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.args = args
        self.client_id = client_id
        self.class_number = args.data.class_number
        self.adapt_iters = 1
        self.model = get_model(args)
        if 'tiny' in args.data.dataset:
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 200)

    def init_weight(self, ckpt):
        # load state dict
        self.model.load_state_dict(ckpt)
        self.model_state = ckpt
        self.model_anchor = copy.deepcopy(self.model)

        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)

        self.past_per_model = None
        if self.args.other.method == 'moon':
            self.glob_model = None
            # The variable to store the previous models.
            self.prev_models = []
            # The max length of prev_models
            self.queue_len = 1

    def preprocess_data(self, data):
        return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def test_source(self, test_data):
        self.adapt_loader = DataLoader(test_data, batch_size=self.args.learn.batch_size, shuffle=False)

        self.model_anchor.eval()
        self.model_anchor.to(self.device)
        self.model_anchor.requires_grad_(False)

        self.sample_num = len(test_data)
        monitor = VariableMonitor()
        criterion = nn.CrossEntropyLoss()
        # Main test loop.
        with torch.no_grad():
            for _, data in enumerate(self.adapt_loader):
                preprocessed_data = self.preprocess_data(data)
                # Update total sample number.
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

                feature, out = self.model_anchor(batch_x, mode='compute-feature-logit')
                y_pred = torch.argmax(out, dim=-1)

                feature_mean = feature.mean(dim=0)
                feature_indicator = feature_mean

                loss = criterion(out, batch_y)
                monitor.append(
                    {
                        'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                        'test_loss': loss.item()
                    },
                    weight=preprocessed_data['y'].shape[0]
                )

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')
        return mean_monitor_variables, feature_indicator

    def _store_prev_model(self, model: nn.Module) -> None:
        r"""
        Overview:
            Store the prev model for fedmoon loss calculation.
        """
        if len(self.prev_models) >= self.queue_len:
            self.prev_models.pop(0)
        self.prev_models.append(copy.deepcopy(model))

    def adapt(self, test_data, device=None, rst=0.01):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)
        if 'abc' in self.args.data.dataset:
            self.model.requires_grad_(False)
            params, names = [], []
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
        else:
            self.model.requires_grad_(False)
            params, names = [], []
            for np, p in self.model.named_parameters():
                if 'fc' not in np:
                    p.requires_grad_(True)
                    params.append(p)
                    names.append(f"{np}")

        # self.adapt_loader = DataLoader(test_data, batch_size=self.args.learn.batch_size, shuffle=False)
        self.sample_num = len(test_data)

        if self.args.other.method == 'moon':
            self.glob_model = copy.deepcopy(self.model)

        self.model_past = copy.deepcopy(self.model)
        self.model_past.cuda()
        flatten_model_past = []
        for param in self.model_past.parameters():
            flatten_model_past.append(param.reshape(-1))
        flatten_model_past = torch.cat(flatten_model_past)

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()
        # Get Local TTA
        beta = 0.95
        theta = 0.5
        for eps in range(1):
            for idx, data in enumerate(self.adapt_loader):
                self.optimizer.zero_grad()
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                z, outputs = self.model(batch_x, mode='compute-feature-logit')

                # (1) entropy
                ent_loss = self.softmax_entropy(outputs).mean(0)

                # (2) diversity
                softmax_out = F.softmax(outputs, dim=-1)
                msoftmax = softmax_out.mean(dim=0)
                ent_loss += torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

               # (3) pseudo label
                # adapt
                py, y_prime = F.softmax(outputs, dim=-1).max(1)
                flag = py > beta
                clf_loss = F.cross_entropy(outputs[flag], y_prime[flag])
                clf_loss = 0
                loss = ent_loss + theta * clf_loss

                if self.args.group.name == 'fedamp_group':
                    for param_p, param in zip(self.model_past.parameters(), self.model.parameters()):
                        loss += (1.0 / 2) * torch.norm((param - param_p) ** 2)
                elif self.args.group.name == 'fedgraph_group':
                    flatten_model = []
                    for param in self.model.parameters():
                        flatten_model.append(param.reshape(-1))
                    flatten_model = torch.cat(flatten_model)
                    loss2 = -0.01 * torch.nn.functional.cosine_similarity(flatten_model.unsqueeze(0),
                                                                  flatten_model_past.unsqueeze(0))
                    loss2.backward()
                elif 'fedprox' in self.args.other.method:
                    lambda_1 = 1.
                    for param_p, param in zip(self.model_past.parameters(), self.model.parameters()):
                        loss += ((lambda_1 / 2) * torch.norm((param - param_p)) ** 2)
                # elif self.args.group.name == 'adapt_group' and self.args.group.aggregation_method == 'st':
                #     flatten_model = []
                #     for param in self.model.parameters():
                #         flatten_model.append(param.reshape(-1))
                #     flatten_model = torch.cat(flatten_model)
                #     loss2 = -0.01 * torch.nn.functional.cosine_similarity(flatten_model.unsqueeze(0),
                #                                                   flatten_model_past.unsqueeze(0))
                #     loss2.backward()
                elif self.args.other.method == 'pfedsd' and self.past_per_model is not None:
                    v_outputs = self.past_per_model(batch_x)
                    KL_temperature = 1.0
                    divergence = F.kl_div(
                        F.log_softmax(outputs / KL_temperature, dim=1),
                        F.softmax(v_outputs / KL_temperature, dim=1),
                        reduction="batchmean",
                    )  # forward KL
                    loss += KL_temperature * KL_temperature * divergence
                elif self.args.other.method == 'moon':
                    temperature = 0.5
                    mu = 1.0
                    # Calculate fedmoon loss.
                    cos = nn.CosineSimilarity(dim=-1)
                    self.glob_model.to(self.device)
                    with torch.no_grad():
                        z_glob, _ = self.glob_model(batch_x, mode='compute-feature-logit')
                    z_i = cos(z, z_glob)
                    logits = z_i.reshape(-1, 1)
                    for prev_model in self.prev_models:
                        prev_model.to(self.device)
                        with torch.no_grad():
                            z_prev, _ = prev_model(batch_x, mode='compute-feature-logit')
                        nega = cos(z, z_prev)
                        logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                    logits /= temperature
                    labels = torch.zeros(batch_x.size(0)).to(self.device).long()
                    fedmoon_loss = criterion(logits, labels)
                    # Add the main loss and fedmoon loss together.
                    loss += mu * fedmoon_loss

                loss.backward()
                self.optimizer.step()

                y_pred = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, batch_y)
                monitor.append(
                    {
                        'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                        'test_loss': loss.item()
                    },
                    weight=preprocessed_data['y'].shape[0]
                )
        if self.args.other.method == 'pfedsd':
            self.past_per_model = copy.deepcopy(self.model)
            self.past_per_model.requires_grad_(False)
        elif self.args.other.method == 'moon':
            self._store_prev_model(self.model)

        # if True:
        #     for nm, m in self.model.named_modules():
        #         for npp, p in m.named_parameters():
        #             if npp in ['weight', 'bias'] and p.requires_grad:
        #                 mask = (torch.rand(p.shape) < rst).float().cuda()
        #                 with torch.no_grad():
        #                     p.data = self.model_state[f"{nm}.{npp}"].cuda() * mask + p * (1. - mask)

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')
        return mean_monitor_variables

    def inference(self, classifier=None, device=None):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

        criterion = nn.CrossEntropyLoss()

        monitor = VariableMonitor()

        with torch.no_grad():

            for _, data in enumerate(self.adapt_loader):
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                outputs = self.model(batch_x)

                y_pred = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, batch_y)
                monitor.append(
                    {
                        'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                        'test_loss': loss.item()
                    },
                    weight=preprocessed_data['y'].shape[0]
                )

        mean_monitor_variables = monitor.variable_mean()
        self.model.to('cpu')
        return mean_monitor_variables







