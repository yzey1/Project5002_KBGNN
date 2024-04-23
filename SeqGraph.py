import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class SeqGraph(MessagePassing):
    """
    Sequence Graph Neural Network model that performs message passing on the graph.
    Include random walk graph neural network model implementation. (Reference: https://github.com/giannisnik/rwgnn/tree/main)

    Args:
        max_step (int): The maximum number of propagation steps.
        hidden_dim (int): The dimensionality of the hidden features.
        hidden_graph_num (int): The number of hidden graphs.
        hidden_graph_size (int): The size of each hidden graph.
        device (torch.device): The device on which the model will be run.

    Attributes:
        max_step (int): The maximum number of propagation steps.
        device (torch.device): The device on which the model will be run.
        hid_dim (int): The dimensionality of the hidden features.
        hidden_graph_num (int): The number of hidden graphs.
        hidden_graph_size (int): The size of each hidden graph.
        fc (torch.nn.Linear): The fully connected layer.
        hidden_adj (torch.nn.Parameter): The parameter for hidden adjacency matrix.
        hidden_feat (torch.nn.Parameter): The parameter for hidden features.
        bn (torch.nn.BatchNorm1d): The batch normalization layer.
        fc1 (torch.nn.Linear): The first fully connected layer.
        fc2 (torch.nn.Linear): The second fully connected layer.
        dropout (torch.nn.Dropout): The dropout layer.
        relu (torch.nn.LeakyReLU): The LeakyReLU activation function.
        sigmoid (torch.nn.Sigmoid): The sigmoid activation function.

    Methods:
        _init_weights(): Initializes the weights of the model.
        forward(data, poi_embeds): Performs forward pass through the model.

    """

    def __init__(self, max_step, hidden_dim, hidden_graph_num, hidden_graph_size, device):
        super(SeqGraph, self).__init__()
        self.max_step = max_step
        self.device = device
        self.hid_dim = hidden_dim
        self.hidden_graph_num = hidden_graph_num
        self.hidden_graph_size = hidden_graph_size

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_adj = nn.Parameter(torch.empty(hidden_graph_num, (hidden_graph_size *(hidden_graph_size - 1)) // 2))
        self.hidden_feat = nn.Parameter(torch.empty(hidden_graph_num, hidden_graph_size, hidden_dim))
        
        self.bn = nn.BatchNorm1d(hidden_graph_num * self.max_step)
        self.fc1 = torch.nn.Linear(hidden_graph_num * self.max_step, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.hidden_adj)
        nn.init.xavier_normal_(self.hidden_feat)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, data, poi_embeds):
        
        # adjacency matrix for hidden graphs
        adj_hidden_norm = torch.zeros(self.hidden_graph_num, self.hidden_graph_size, self.hidden_graph_size).to(self.device)
        idx = torch.triu_indices(self.hidden_graph_size, self.hidden_graph_size, 1)
        adj_hidden_norm[:, idx[0], idx[1]] = self.relu(self.hidden_adj)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        
        # get the features and adjacency matrix of the POIs
        poi_feat = poi_embeds(data.x.squeeze())
        poi_adj = data.edge_index
        graph_indicator = data.batch
        unique = torch.unique(graph_indicator)
        n_graphs = unique.size(0)
        
        # apply fully connected layer and sigmoid activation function
        x = self.sigmoid(self.fc(poi_feat))
        z = self.hidden_feat
        zx = torch.einsum("abc,dc->abd", (z, x))

        out = []
        for i in range(self.max_step):
            if i == 0:
                # initialize the hidden graph features
                eye = torch.eye(self.hidden_graph_size, device=self.device)
                eye = eye.repeat(self.hidden_graph_num, 1, 1)
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dc->abd", (o, x))
            else:
                # propagate the features
                x = self.propagate(poi_adj, x=x, size=None)
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dc->abd", (z, x))
            
            # apply dropout and concatenate the features
            t = self.dropout(t)
            t = torch.mul(zx, t)
            
            # sum the features and transpose
            t = torch.zeros(t.size(0), t.size(1), n_graphs,
                            device=self.device).index_add_(2, graph_indicator, t)
            t = torch.sum(t, dim=1)
            t = torch.transpose(t, 0, 1)
            
            # append the features to the output
            out.append(t)
            
        # concatenate the output and apply fully connected layers
        out = torch.cat(out, dim=1)
        out = self.bn(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
