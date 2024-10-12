import torch
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool, EdgePooling, global_max_pool
from torch.nn import Linear, BatchNorm1d, Module, ReLU
from Initial_Training.Models.isomorphism import GraphIsomorphismNetwork
import torch.nn.functional as F


class DrugRegressionModel(Module):
    def __init__(self, num_labels, encoder):
        super(DrugRegressionModel, self).__init__()

        self.nn = GraphIsomorphismNetwork(encoder=encoder)
        self.linear1 = Linear(in_features=512, out_features=256)
        self.attack_attention = Linear(in_features=256, out_features=3)
        self.classifier = Linear(in_features=3, out_features=num_labels)

        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=256)

    def forward(self, v, attack_vector, edges, batch):
        v = self.nn.forward(v, edges)
        v = global_mean_pool(v, batch=batch)  # Graph View

        v = self.linear1(v)
        v = self.bn(v)
        v = self.relu(v)

        v = self.attack_attention(v)
        v = torch.mul(attack_vector, v)

        v = self.classifier(v)

        return v


class DrugClassificationModel(Module):
    def __init__(self, num_labels, encoder):
        super(DrugClassificationModel, self).__init__()

        self.nn = GraphIsomorphismNetwork(encoder=encoder)
        self.linear1 = Linear(in_features=512, out_features=256)
        self.attack_attention = Linear(in_features=256, out_features=3)
        self.classifier = Linear(in_features=3, out_features=num_labels)

        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=256)

    def forward(self, v, attack_vector, edges, batch):
        v = self.nn.forward(v, edges)
        v = global_mean_pool(v, batch=batch)  # Graph View

        v = self.linear1(v)
        v = self.bn(v)
        v = self.relu(v)

        v = self.attack_attention(v)
        v = torch.mul(attack_vector, v)

        v = self.classifier(v)

        return v, F.sigmoid(v)
