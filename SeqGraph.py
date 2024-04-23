import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class RW_NN(MessagePassing):
    def __init__(self, max_step, hidden_dim, hidden_graph_num, hidden_graph_size, device):
        super(RW_NN, self).__init__()
        self.max_step = max_step
        self.device = device
        self.hid_dim = hidden_dim
        self.hidden_graph_num = hidden_graph_num
        self.hidden_graph_size = hidden_graph_size

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_adj = nn.Parameter(torch.empty(hidden_graph_num, (hidden_graph_size *(hidden_graph_size - 1)) // 2))
        self.hidden_feat = nn.Parameter(torch.empty(hidden_graph_num, hidden_graph_size, hidden_dim))
        
        self.bn = nn.BatchNorm1d(hidden_graph_num * self.max_step)
        self.mlp = torch.nn.Linear(hidden_graph_num * self.max_step, hidden_dim)

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
        poi_feat, poi_adj = poi_embeds(data.x.squeeze()), data.edge_index
        graph_indicator = data.batch
        unique = torch.unique(graph_indicator)
        n_graphs = unique.size(0)

        adj_hidden_norm = torch.zeros(self.hidden_graph_num, self.hidden_graph_size, self.hidden_graph_size).to(self.device)
        idx = torch.triu_indices(self.hidden_graph_size, self.hidden_graph_size, 1)
        adj_hidden_norm[:, idx[0], idx[1]] = self.relu(self.hidden_adj)
        adj_hidden_norm = adj_hidden_norm + \
            torch.transpose(adj_hidden_norm, 1, 2)
        x = self.sigmoid(self.fc(poi_feat))
        z = self.hidden_feat
        zx = torch.einsum("abc,dc->abd", (z, x))

        out = []
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.hidden_graph_size, device=self.device)
                eye = eye.repeat(self.hidden_graph_num, 1, 1)
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dc->abd", (o, x))
            else:
                x = self.propagate(poi_adj, x=x, size=None)
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dc->abd", (z, x))
            t = self.dropout(t)
            t = torch.mul(zx, t)
            t = torch.zeros(t.size(0), t.size(1), n_graphs,
                            device=self.device).index_add_(2, graph_indicator, t)
            t = torch.sum(t, dim=1)
            t = torch.transpose(t, 0, 1)
            out.append(t)

        out = torch.cat(out, dim=1)
        out = self.relu(self.mlp(out))
        out = self.dropout(out)
        return out


class SeqGraph(nn.Module):
    def __init__(self, max_step, embed_dim, hidden_graph_num, hidden_graph_size, device):
        super(SeqGraph, self).__init__()
        self.rwnn = RW_NN(max_step, embed_dim, hidden_graph_num, hidden_graph_size, device)

    def forward(self, data, poi_embeds):
        sess_feat = self.rwnn(data, poi_embeds)
        return sess_feat
