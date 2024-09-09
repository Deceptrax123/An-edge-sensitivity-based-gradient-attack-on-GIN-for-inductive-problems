import torch
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool, EdgePooling, global_max_pool
from torch.nn import Linear, BatchNorm1d, Module
import torch.nn.functional as F

# Same model architecture for all problems


class DrugClassificationModel(Module):
    def __init__(self, input_features, num_features, num_labels, category):
        super(DrugClassificationModel, self).__init__()
        self.category = category

        self.initial = GCNConv(in_channels=input_features,
                               out_channels=1, normalize=True)
        self.layer1 = GCNConv(in_channels=1,
                              out_channels=num_features*2, normalize=True)
        self.layer2 = GCNConv(in_channels=num_features*2,
                              out_channels=num_features*4, normalize=True)
        self.layer3 = GCNConv(in_channels=num_features*4,
                              out_channels=num_features*8, normalize=True)

        self.linear = Linear(in_features=num_features*8,
                             out_features=num_features*16)
        self.bn = BatchNorm1d(num_features*16)

        self.classifier = Linear(
            in_features=num_features*16, out_features=num_labels)

    def forward(self, v, edges, batch):
        v = self.initial(v, edges)
        v = F.relu(self.layer1(v, edges))
        v = F.relu(self.layer2(v, edges))
        v = F.relu(self.layer3(v, edges))

        v1 = global_mean_pool(v, batch)
        v2 = global_max_pool(v, batch)
        v = torch.add(v1, v2)

        v = self.linear(v)
        v = self.bn(v)

        v = self.classifier(v)

        if self.category == 'binary' or self.category == 'multilabel':
            return v, F.sigmoid(v)
        else:
            return v, F.softmax(v)


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


class NodeClassificationModel(Module):
    def __init__(self, num_features, num_labels):
        super(NodeClassificationModel, self).__init__()

        self.layer1 = GCNConv(in_channels=num_features,
                              out_channels=num_features*2)
        self.gn1 = GraphNorm(num_features*2)

        self.layer2 = GCNConv(in_channels=num_features*2,
                              out_channels=num_features*4)
        self.gn2 = GraphNorm(num_features*4)

        self.layer3 = GCNConv(in_channels=num_features*4,
                              out_channels=num_features*8)
        self.gn3 = GraphNorm(num_features*8)

        self.classifier = GCNConv(
            in_channels=num_features*8, out_channels=num_labels)

    def forward(self, v, edges):
        v = F.relu(self.layer1(v, edges))
        v = self.gn1(v)

        v = F.relu(self.layer2(v, edges))
        v = self.gn2(v)

        v = F.relu(self.layer3(v, edges))
        v = self.gn3(v)

        v = self.classifier(v)

        return v, F.softmax(v)
