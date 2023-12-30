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

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for eps in range(1):
            monitor = VariableMonitor()
            for _, data in enumerate(self.adapt_loader):
                self.optimizer.zero_grad()

                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                outputs = self.model(batch_x)
                y_pred = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, y_pred)

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

    def inference(self, classifier=None, device=None, ap=0.72, mt=0.99, rst=0.1):
        if device is not None:
            self.device = device
        self.model.to(self.device)
        if self.args.other.is_average:
            self.model_ema = self.update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=mt)

        self.model.eval()
        self.model.requires_grad_(False)

        criterion = nn.CrossEntropyLoss()
        monitor = VariableMonitor()

        with torch.no_grad():
            for _, data in enumerate(self.adapt_loader):
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

                outputs = self.model(batch_x)

                # Teacher Prediction
                anchor_prob = torch.nn.functional.softmax(self.model_anchor(batch_x), dim=1).max(1)[0]
                standard_ema = self.model_ema(batch_x)
                # Augmentation-averaged Prediction
                N = 32
                outputs_emas = []
                for i in range(N):
                    outputs_ = self.model_ema(self.transform(batch_x)).detach()
                    outputs_emas.append(outputs_)
                # # Threshold choice discussed in supplementary
                if anchor_prob.mean(0) < ap:
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                # Student update
                y_pred = torch.argmax(outputs_ema, dim=-1)
                # y_pred = torch.argmax(outputs, dim=-1)
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



