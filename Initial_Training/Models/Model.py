import torch
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool, EdgePooling, global_max_pool
from torch.nn import Linear, BatchNorm1d, Module, ReLU
from Initial_Training.Models.isomorphism import GraphIsomorphismNetwork
import torch.nn.functional as F

# Same model architecture for all problems


class DrugRegressionModel(Module):
    def __init__(self, num_labels, encoder):
        super(DrugRegressionModel, self).__init__()

        self.nn = GraphIsomorphismNetwork(encoder=encoder)
        self.linear1 = Linear(in_features=512, out_features=256)
        self.classifier = Linear(in_features=256, out_features=num_labels)

        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=256)

    def forward(self, v, edges, batch):
        v = self.nn.forward(v, edges)
        v = global_mean_pool(v, batch=batch)  # Graph View

        v = self.linear1(v)
        v = self.bn(v)
        v = self.relu(v)

        v = self.classifier(v)

        return v


class DrugClassificationModel(Module):
    def __init__(self, num_labels, encoder):
        super(DrugClassificationModel, self).__init__()

        self.nn = GraphIsomorphismNetwork(encoder=encoder)
        self.linear1 = Linear(in_features=512, out_features=256)
        self.classifier = Linear(in_features=256, out_features=num_labels)

        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=256)

    def forward(self, v, edges, batch):
        v = self.nn.forward(v, edges)
        v = global_mean_pool(v, batch=batch)  # Graph View

        v = self.linear1(v)
        v = self.bn(v)
        v = self.relu(v)

        v = self.classifier(v)

        return v, F.sigmoid(v)


class GraphClassificationModel(Module):
    def __init__(self, num_features, num_labels, category):
        super(GraphClassificationModel, self).__init__()
        self.category = category
        self.layer1 = GCNConv(in_channels=num_features,
                              out_channels=num_features*2)
        self.gn1 = GraphNorm(num_features*2)
        self.edge1 = EdgePooling(
            in_channels=num_features*2, dropout=0.2, add_to_edge_score=0.5)

        self.layer2 = GCNConv(in_channels=num_features*2,
                              out_channels=num_features*4)
        self.gn2 = GraphNorm(num_features*4)
        self.edge2 = EdgePooling(
            in_channels=num_features*4, dropout=0.2, add_to_edge_score=0.5)

        self.layer3 = GCNConv(in_channels=num_features*4,
                              out_channels=num_features*8)
        self.gn3 = GraphNorm(num_features*8)
        self.edge3 = EdgePooling(
            in_channels=num_features*8, dropout=0.2, add_to_edge_score=0.5)

        self.linear = Linear(in_features=num_features*8,
                             out_features=num_features*16)
        self.bn = BatchNorm1d(num_features*16)

        self.classifier = Linear(
            in_features=num_features*16, out_features=num_labels)

    def forward(self, v, edges, batch):
        v = F.relu(self.layer1(v, edges))
        v = self.gn1(v)
        v, edges, batch, _ = self.edge1(v, edges, batch)

        v = F.relu(self.layer2(v, edges))
        v = self.gn2(v)
        v, edges, batch, _ = self.edge2(v, edges, batch)

        v = F.relu(self.layer3(v, edges))
        v = self.gn3(v)
        v, edges, batch, _ = self.edge3(v, edges, batch)

        v1 = global_mean_pool(v, batch)
        v2 = global_max_pool(v, batch)
        v = torch.add(v1, v2)

        v = self.linear(v)
        v = self.bn(v)

        v = self.classifier(v)

        if self.category == 'binary' or self.category == 'multilabel':
            return v, F.sigmoid(v)
        else:
            return v, F.softmax(v, dim=1)
