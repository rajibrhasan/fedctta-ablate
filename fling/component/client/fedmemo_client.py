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
from fling.dataset.aug_data import aug
import fling.component.client.my_transformers as my_transforms
import PIL

def get_normalization(data_name):
    if "cifar10" in data_name:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.247, 1/0.243, 1/0.261]),
                                          transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1., 1., 1.])])
        return normalize, unnormalize
    elif "cifar100" in data_name:
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.2675, 1/0.2565, 1/0.2761]),
                                          transforms.Normalize(mean=[-0.5071, -0.4867, -0.4408], std=[1., 1., 1.])])
        return normalize, unnormalize
    elif "imagenet" in data_name:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        unnormalize = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
                                          transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.])])
        return normalize, unnormalize

@CLIENT_REGISTRY.register('fedmemo_client')
class FedMEMOClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedMEMOClient, self).__init__(args, client_id, train_dataset, test_dataset)
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

    def marginal_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

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

        self.model.train()
        # turn on model grads.
        self.model.requires_grad_(True)

        # do the unnormalize to ensure consistency.
        normalize, unnormalize = get_normalization(self.args.data.dataset)
        convert_img = transforms.Compose([unnormalize, transforms.ToPILImage()])
        model_param = copy.deepcopy(self.model.state_dict())
        monitor = VariableMonitor()
        num_steps = 1
        for _, data in enumerate(self.adapt_loader):
            preprocessed_data = self.preprocess_data(data)
            batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
            for i in range(batch_x.shape[0]):
                image = convert_img(batch_x[i])
                for _ in range(num_steps):
                    # generate a batch of augmentations and minimize prediction entropy.
                    inputs = [aug(image, normalize) for _ in range(16)]
                    inputs = torch.stack(inputs).cuda()
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss, avg_outputs = self.marginal_entropy(outputs)

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
                        lambda_1 = 0.01
                        for param_p, param in zip(self.model_past.parameters(), self.model.parameters()):
                            loss += ((lambda_1 / 2) * torch.norm((param - param_p)) ** 2)

                    loss.backward()
                    self.optimizer.step()
                    y_pred = torch.argmax(avg_outputs, dim=-1)

                    monitor.append(
                        {
                            'test_acc': torch.mean((y_pred == batch_y).float()).item(),
                            'test_loss': loss.item()
                        },
                        weight=batch_y.shape[0]
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



