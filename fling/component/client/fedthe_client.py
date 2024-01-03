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
import torchvision.transforms as transforms
from fling.dataset.aug_data import aug
import numpy as np

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

@CLIENT_REGISTRY.register('fedthe_client')
class FedTHEClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedTHEClient, self).__init__(args, client_id, train_dataset, test_dataset)
        self.args = args
        self.client_id = client_id
        self.class_number = args.data.class_number
        self.adapt_iters = 1
        self.model = get_model(args)
        self.test_history = None

    def init_weight(self, ckpt):
        # load state dict
        self.model.load_state_dict(ckpt)
        self.model_state = ckpt
        self.model_anchor = copy.deepcopy(self.model)

        self.model.requires_grad_(True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)

    def init_globalrep(self, global_rep):
        self.global_rep = global_rep

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

    def marginal_entropy(self, outputs):
        logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
        avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
        min_real = torch.finfo(avg_logits.dtype).min
        avg_logits = torch.clamp(avg_logits, min=min_real)
        return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

    def _test_time_tune(self, num_steps=3):
        # turn on model grads.
        self.model.requires_grad_(True)
        self.personalized_head.requires_grad_(True)
        # set optimizer.
        fe_optim = torch.optim.SGD(self.model.parameters(), lr=0.00001)
        fe_optim.add_param_group({"params": self.personalized_head.parameters()})
        g_pred, p_pred = [], []
        # do the unnormalize to ensure consistency.
        normalize, unnormalize = get_normalization(self.args.data.dataset)
        convert_img = transforms.Compose([unnormalize, transforms.ToPILImage()])
        agg_softmax = torch.nn.functional.softmax(self.agg_weight).detach()
        model_param = copy.deepcopy(self.model.state_dict())
        p_head_param = copy.deepcopy(self.personalized_head.state_dict())

        for _, data in enumerate(self.adapt_loader):
            preprocessed_data = self.preprocess_data(data)
            batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
            for i in range(batch_x.shape[0]):
                image = convert_img(batch_x[i])
                for _ in range(num_steps):
                    # generate a batch of augmentations and minimize prediction entropy.
                    inputs = [aug(image, normalize) for _ in range(16)]
                    inputs = torch.stack(inputs).cuda()
                    fe_optim.zero_grad()
                    feature, g_out = self.model(inputs, mode='compute-feature-logit')
                    p_out = self.personalized_head(feature)
                    agg_output = agg_softmax[i, 0] * g_out + agg_softmax[i, 1] * p_out
                    loss, _ = self.marginal_entropy(agg_output)
                    loss.backward()
                    fe_optim.step()
                with torch.no_grad():
                    feature, g_out = self.model(batch_x[i].unsqueeze(0).cuda(), mode='compute-feature-logit')
                    p_out = self.personalized_head(feature)
                    g_pred.append(g_out)
                    p_pred.append(p_out)
                self.model.load_state_dict(model_param)
                self.personalized_head.load_state_dict(p_head_param)

        # turn off grads.
        self.model.requires_grad_(False)
        self.personalized_head.requires_grad_(False)
        return torch.cat(g_pred), torch.cat(p_pred)

    def adapt(self, test_data, device=None):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)
        self.model.requires_grad_(False)

        # get personalized head
        self.personalized_head = copy.deepcopy(self.model.fc)
        self.personalized_head.to(self.device)
        self.personalized_head.requires_grad_(True)

        criterion = nn.CrossEntropyLoss()
        num_epochs = 10
        p_optimizer = torch.optim.Adam(self.personalized_head.parameters(), lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)
        for eps in range(num_epochs):
            for _, data in enumerate(self.adapt_loader):
                p_optimizer.zero_grad()
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
                feature, _ = self.model(batch_x, mode='compute-feature-logit')
                p_output = self.personalized_head(feature)
                loss = self.softmax_entropy(p_output).mean(0)
                loss.backward()
                p_optimizer.step()

        # calculate aggregation weight
        temperature = torch.hstack((torch.ones((feature.shape[0], 1)).cuda(), torch.ones((feature.shape[0], 1)).cuda()))
        self.agg_weight = torch.nn.Parameter(torch.tensor(temperature).cuda(), requires_grad=True)
        e_optimizer = torch.optim.Adam([self.agg_weight], lr=self.args.learn.optimizer.lr)

        self.model.requires_grad_(False)
        self.personalized_head.requires_grad_(False)

        alpha = 0.1
        for _, data in enumerate(self.adapt_loader):
            preprocessed_data = self.preprocess_data(data)
            batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']
            feature, g_out = self.model(batch_x, mode='compute-feature-logit')
            p_out = self.personalized_head(feature)

            test_history = None
            for i in range(feature.shape[0]):
                if test_history is None and self.test_history is None:
                    test_history = [feature[0, :]]
                elif test_history is None and self.test_history is not None:
                    test_history = [self.test_history[-1, :]]
                else:
                    test_history.append(alpha * feature[i, :] + (1 - alpha) * test_history[-1])
            self.test_history = torch.stack(test_history)
            self.local_rep = torch.mean(feature, dim=0)

        beta = 0.1
        for _ in range(num_epochs):
            e_optimizer.zero_grad()
            # normalize the aggregation weight by softmax
            agg_softmax = torch.nn.functional.softmax(self.agg_weight, dim=1)
            agg_output = agg_softmax[:, 0].unsqueeze(1) * g_out.detach() \
                         + agg_softmax[:, 1].unsqueeze(1) * p_out.detach()
            # formulate test representation.
            test_rep = beta * feature + (1 - beta) * self.test_history
            p_feat_al = torch.norm((test_rep - self.local_rep), dim=1)
            g_feat_al = torch.norm((test_rep - self.global_rep), dim=1)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(F.softmax(g_out).detach(), F.softmax(p_out).detach())
            # loss function based on prediction similarity, entropy minimization and feature alignment.
            loss = (-sim * (agg_output.softmax(1) * agg_output.log_softmax(1)).sum(1) + \
                (1 - sim) * (agg_softmax[:, 0] * g_feat_al.detach() + agg_softmax[:, 1] * p_feat_al.detach())).mean(0)
            loss.backward()

            if torch.norm(self.agg_weight.grad) < 1e-5:
                break
            e_optimizer.step()

        g_pred, p_pred = self._test_time_tune(num_steps=3)
        # inference procedure for multi-head nets.
        agg_softmax = torch.nn.functional.softmax(self.agg_weight)
        agg_output = agg_softmax[:, 0].unsqueeze(1) * g_pred \
                     + agg_softmax[:, 1].unsqueeze(1) * p_pred
        y_pred = torch.argmax(agg_output, dim=-1)
        # evaluate the output and get the loss, performance.
        loss = criterion(agg_output, batch_y)
        monitor = VariableMonitor()
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



