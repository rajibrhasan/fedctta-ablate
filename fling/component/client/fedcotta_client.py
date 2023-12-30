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

@CLIENT_REGISTRY.register('fedcotta_client')
class FedCoTTAClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedCoTTAClient, self).__init__(args, client_id, train_dataset, test_dataset)
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
        self.model.requires_grad_(False)
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)
        # collect_params
        names = []
        params = []
        for nm, m in self.model.named_modules():
            if True:
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        self.optimizer = torch.optim.Adam(params, lr=self.args.learn.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.)
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

    def get_tta_transforms(self, gaussian_std: float = 0.005, soft=False, clip_inputs=False):
        img_shape = (32, 32, 3)
        n_pixels = img_shape[0]
        clip_min, clip_max = 0.0, 1.0
        p_hflip = 0.5
        tta_transforms = transforms.Compose([
            my_transforms.Clip(0.0, 1.0),
            my_transforms.ColorJitterPro(
                brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                gamma=[0.85, 1.15] if soft else [0.7, 1.3]
            ),
            transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
            transforms.RandomAffine(
                degrees=[-8, 8] if soft else [-15, 15],
                translate=(1 / 16, 1 / 16),
                scale=(0.95, 1.05) if soft else (0.9, 1.1),
                shear=None,
                # resample=PIL.Image.BILINEAR,
                interpolation=PIL.Image.BILINEAR,
                # fillcolor=None
                fill=None
            ),
            transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
            transforms.CenterCrop(size=n_pixels),
            transforms.RandomHorizontalFlip(p=p_hflip),
            my_transforms.GaussianNoise(0, gaussian_std),
            my_transforms.Clip(clip_min, clip_max)
        ])
        return tta_transforms

    def softmax_entropy_cotta(self, x, x_ema):  # -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

    def update_ema_variables(self, ema_model, model, alpha_teacher):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        return ema_model

    def adapt(self, test_data, device=None, ap=0.72, mt=0.999, rst=0.01):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)
        self.model_anchor.to(self.device)
        self.model_anchor.requires_grad_(False)
        self.model_ema = copy.deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        self.model_ema.to(self.device)
        self.model_ema.requires_grad_(False)
        self.model.requires_grad_(False)
        # enable all params trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)

        self.transform = self.get_tta_transforms()

        self.model.train()

        for eps in range(1):
            monitor = VariableMonitor()
            for _, data in enumerate(self.adapt_loader):
                self.optimizer.zero_grad()
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
                # Threshold choice discussed in supplementary
                if anchor_prob.mean(0) < ap:
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema
                # Student update
                y_pred = torch.argmax(outputs_ema, dim=-1)
                # if self.args.other.loss == 'sup':
                #     loss = criterion(outputs, batch_y)
                # else:
                #     loss = criterion(outputs, y_pred)
                loss = (self.softmax_entropy_cotta(outputs, outputs_ema)).mean(0)
                loss.backward()
                self.optimizer.step()
                # Teacher update
                self.model_ema = self.update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=mt)
                # Stochastic restore
                if True:
                    for nm, m in self.model.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                mask = (torch.rand(p.shape) < rst).float().cuda()
                                with torch.no_grad():
                                    p.data = self.model_state[f"{nm}.{npp}"].cuda() * mask + p * (1. - mask)

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



