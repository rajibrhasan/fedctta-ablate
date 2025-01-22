import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Iterable

from fling.utils.registry_utils import CLIENT_REGISTRY
from .client_template import ClientTemplate
from fling.model import get_model
from fling.utils.utils import VariableMonitor
from sklearn.decomposition import PCA


@CLIENT_REGISTRY.register('fedtent_client')
class FedTentClient(ClientTemplate):

    def __init__(self, args: dict, client_id: int, train_dataset: Iterable = None, test_dataset: Iterable = None):
        super(FedTentClient, self).__init__(args, client_id, train_dataset, test_dataset)
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

                # feature_mean = feature.mean(dim=0)
                # feature_indicator = feature_mean

                if self.args.other.feat_sim == 'feature':
                    feature_mean = feature.mean(dim=0)
                    feature_indicator = feature_mean

                elif self.args.other.feat_sim == 'pvec':
                    n_samples = batch_x.size(0)
                    X_flat = batch_x.view(n_samples, -1).cpu().numpy()
                    pca = PCA(n_components=2)
                    X_reduced = pca.fit_transform(X_flat)
                    components = torch.tensor(pca.components_).flatten().to(self.device)
                    feature_indicator = components
                
                


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

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def adapt(self, test_data, device=None):
        if device is not None:
            device_bak = self.device
            self.device = device
        self.model.to(self.device)
        self.sample_num = len(test_data)

        self.model_past = copy.deepcopy(self.model)
        self.model_past.cuda()
        flatten_model_past = []
        for param in self.model_past.parameters():
            flatten_model_past.append(param.reshape(-1))
        flatten_model_past = torch.cat(flatten_model_past)

        # Turn on model grads. collect_params
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
        self.model.train()
        # Get Local TTA
        monitor = VariableMonitor()

        for eps in range(self.adapt_iters):
            for idx, data in enumerate(self.adapt_loader):
                self.optimizer.zero_grad()
                preprocessed_data = self.preprocess_data(data)
                batch_x, batch_y = preprocessed_data['x'], preprocessed_data['y']

                out = self.model(batch_x)
                y_pred = torch.argmax(out, dim=-1)

                loss = self.softmax_entropy(out).mean(0)

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

                monitor.append(
                    {
                        'test_acc': torch.mean((y_pred == preprocessed_data['y']).float()).item(),
                        'test_loss': loss.item()
                    },
                    weight=preprocessed_data['y'].shape[0]
                )
                loss.backward()
                self.optimizer.step()

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



