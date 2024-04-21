import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import degree


class SelfAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)

    def forward(self, sess_embed, sections):
        v_i = torch.split(sess_embed, sections)
        v_i_pad = pad_sequence(v_i, batch_first=True, padding_value=0.)
        # v_i = torch.stack(v_i)

        attn_output, _ = self.multihead_attn(v_i_pad, v_i_pad, v_i_pad)

        return attn_output


class Geo_GCN(nn.Module):
    # message calculation between two nodes, equation(2)
    """
    compute the message from one node to its neighboring node
    equation(2) in Methodology part B 

    Args:
        x: POI representations from last layer
        edge_index: the index of source node and destination node of the edge
        dist_vec: the vector of the distance between two nodes

    return:
        self.W(side_embed): the computed message
    """
    # initialize the linear transformation

    def __init__(self, in_channels, out_channels, device):
        super(Geo_GCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels).to(device)
        self.init_weights()

    # initialize the weight matrices
    def init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)

    # compute the message from one POI to its neighbors
    def forward(self, x, edge_index, dist_vec):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        dist_weight = torch.exp(-(dist_vec ** 2))
        dist_adj = torch.sparse_coo_tensor(edge_index, dist_weight * norm)
        side_embed = torch.sparse.mm(dist_adj, x)
        # make linear transformation with the weight matrice
        return self.W(side_embed)


# GeoGraph Neural Network
class GeoGraph(nn.Module):
    def __init__(self, n_poi, gcn_num, embed_dim, dist_edges, dist_vec, num_heads, device):
        super(GeoGraph, self).__init__()
        self.n_poi = n_poi
        self.embed_dim = embed_dim
        self.gcn_num = gcn_num
        self.device = device

        self.dist_edges = dist_edges.to(device)
        loop_index = torch.arange(0, n_poi).unsqueeze(
            0).repeat(2, 1).to(device)
        self.dist_edges = torch.cat(
            (self.dist_edges, self.dist_edges[[1, 0]], loop_index), dim=-1)

        # update distance vector tensor to match with dist_edges
        dist_vec = np.concatenate((dist_vec, dist_vec, np.zeros(self.n_poi)))
        self.dist_vec = torch.Tensor(dist_vec).to(device)

        # define a sequential model with two linear transformations
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # initialize GCN module, selfAttn, weights
        self.gcn = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.gcn.append(Geo_GCN(embed_dim, embed_dim, device).to(device))

        self.init_weights()

        self.selfAttn = SelfAttn(self.embed_dim, num_heads).to(device)

    # Initialize weights in the model
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, data, poi_embeds):
        batch_idx = data.batch  # batch index
        seq_lens = torch.bincount(batch_idx)   # number of POIs in the batch
        sections = tuple(seq_lens.cpu().numpy())  # transform to tuple
        enc = poi_embeds.embeds.weight  # get the embeddings of POI
        for i in range(self.gcn_num):
            enc = self.gcn[i](enc, self.dist_edges, self.dist_vec)
            enc = F.leaky_relu(enc)
            enc = F.normalize(enc, dim=-1)

        # embeddings of the target nodes
        tar_embed = enc[data.poi]
        # embeddings of other nodes in the graph
        geo_feat = enc[data.x.squeeze()]

        # apply multihead self-attention
        self_attn_feat = self.selfAttn(geo_feat, sections)
        # aggregate self-attention features to obtain semantic representation e_g,u
        aggr_feat = torch.mean(self_attn_feat, dim=1)

        # proj_head(graph_enc) looks like e_g,u, however it does not apply multi-head self-attention
        # return self.proj_head(graph_enc), tar_embed
        return aggr_feat, tar_embed
