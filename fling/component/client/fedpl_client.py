import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Iterable

from fling.utils.registry_utils import CLIENT_REGISTRY
from .client_template import ClientTemplate
from fling.model import get_model
from fling.utils.utils import VariableMonitor

import torchvision.transforms as transforms
import fling.component.client.my_transformers as my_transforms
import PIL

@CLIENT_REGISTRY.register('fedpl_client')
class FedPLClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedPLClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.args = args
        self.client_id = client_id
        self.class_number = args.data.class_number
        self.adapt_iters = 1
        self.model = get_model(args)

    def init_weight(self, ckpt):
        # load state dict
        self.model.load_state_dict(ckpt)
        self.model_state = ckpt
        self.model_anchor = copy.deepcopy(self.model)

        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)

    def preprocess_data(self, data):
        return {'x': data['input'].to(self.device), 'y': data['class_id'].to(self.device)}

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

    def adapt(self, test_data, device=None, ap=0.72, mt=0.999, rst=0.01):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)
        self.model.requires_grad_(True)

        self.model_past = copy.deepcopy(self.model)
        self.model_past.cuda()
        flatten_model_past = []
        for param in self.model_past.parameters():
            flatten_model_past.append(param.reshape(-1))
        flatten_model_past = torch.cat(flatten_model_past)

        criterion = nn.CrossEntropyLoss()
        self.model.train()
        monitor = VariableMonitor()
        if self.args.other.method == 'ditto':
            self.local_model = copy.deepcopy(self.model)
            self.local_model.to(self.device)
            self.local_model.requires_grad_(True)
            self.local_model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)
            poptimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)

            lamda = 1.
            num_epochs = 1
            for _ in range(num_epochs):
                for _, data in enumerate(self.adapt_loader):
                    poptimizer.zero_grad()
                    preprocessed_data = self.preprocess_data(data)
                    batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                    outputs = self.local_model(batch_x)
                    y_pred = torch.argmax(outputs, dim=-1)
                    loss = criterion(outputs, y_pred)
                    for param_p, param in zip(self.local_model.parameters(), self.model.parameters()):
                        loss += ((lamda / 2) * torch.norm((param - param_p)) ** 2)
                    monitor.append(
                        {
                            'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                            'test_loss': loss.item()
                        },
                        weight=preprocessed_data['y'].shape[0]
                    )
                    loss.backward()
                    poptimizer.step()

            for _ in range(num_epochs):
                for _, data in enumerate(self.adapt_loader):
                    optimizer.zero_grad()
                    preprocessed_data = self.preprocess_data(data)
                    batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                    outputs = self.local_model(batch_x)
                    y_pred = torch.argmax(outputs, dim=-1)
                    loss = criterion(outputs, y_pred)
                    loss.backward()
                    optimizer.step()
        else:
            for eps in range(1):
                for _, data in enumerate(self.adapt_loader):
                    self.optimizer.zero_grad()

                    preprocessed_data = self.preprocess_data(data)
                    batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                    outputs = self.model(batch_x)
                    y_pred = torch.argmax(outputs, dim=-1)
                    loss = criterion(outputs, y_pred)

                    if self.args.group.name == 'fedamp_group':
                        for param_p, param in zip(self.model_past.parameters(), self.model.parameters()):
                            loss += (1.0 / 2) * torch.norm((param - param_p) ** 2)
                    elif self.args.group.name == 'fedgraph_group':
                        flatten_model = []
                        for param in self.model.parameters():
                            flatten_model.append(param.reshape(-1))
                        flatten_model = torch.cat(flatten_model)
                        loss2 = torch.nn.functional.cosine_similarity(flatten_model.unsqueeze(0), flatten_model_past.unsqueeze(0))
                        loss2.backward()
                    elif 'fedprox' in self.args.other.method:
                        lambda_1 = 1.
                        for param_p, param in zip(self.model_past.parameters(), self.model.parameters()):
                            loss += ((lambda_1 / 2) * torch.norm((param - param_p)) ** 2)
                    elif self.args.group.name == 'adapt_group' and self.args.group.aggregation_method == 'st':
                        flatten_model = []
                        for param in self.model.parameters():
                            flatten_model.append(param.reshape(-1))
                        flatten_model = torch.cat(flatten_model)
                        loss2 = torch.nn.functional.cosine_similarity(flatten_model.unsqueeze(0), flatten_model_past.unsqueeze(0))
                        loss2.backward()

                    loss.backward()
                    self.optimizer.step()

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

    def inference(self, classifier=None, device=None):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()

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



