from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Module
import torch.nn.functional as F

# Same model architecture for all problems


class GlobalModel(Module):
    def __init__(self, num_features, num_labels, category):
        super(GlobalModel, self).__init__()
        self.category = category
        self.layer1 = GCNConv(in_channels=num_features,
                              out_channels=num_features*2)
        self.gn1 = GraphNorm(num_features*2)

        self.layer2 = GCNConv(in_channels=num_features*2,
                              out_channels=num_features*4)
        self.gn2 = GraphNorm(num_features*4)

        self.layer3 = GCNConv(in_channels=num_features*4,
                              out_channels=num_features*8)
        self.gn3 = GraphNorm(num_features*8)

        self.linear = Linear(in_features=num_features*8,
                             out_features=num_features*16)
        self.bn = BatchNorm1d(num_features*16)

        self.classifier = Linear(
            in_features=num_features*16, out_features=num_labels)

    def forward(self, v, edges, batch):
        v = F.relu(self.layer1(v, edges))
        v = self.gn1(v)

        v = F.relu(self.layer2(v, edges))
        v = self.gn2(v)

        v = F.relu(self.layer3(v, edges))
        v = self.gn3(v)

        v = global_mean_pool(v, batch)

        v = self.linear(v)
        v = self.bn(v)

        v = self.classifier(v)

        if self.category == 'binary' or self.category == 'multilabel':
            return v, F.sigmoid(v)
        else:
            return v, F.softmax(v)
