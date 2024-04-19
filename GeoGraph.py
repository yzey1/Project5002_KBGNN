import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import degree


def sequence_mask(lengths, max_len=None):
    # generate a sequence mask, marks the positions of valid elements
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)

    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len, )

    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1))).reshape(lengths_shape)


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
    def __init__(self, in_channels, out_channels, device):
        super(Geo_GCN, self).__init__()
        self.W = nn.Linear(in_channels, out_channels).to(device)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, edge_index, dist_vec):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        dist_weight = torch.exp(-(dist_vec ** 2))
        dist_adj = torch.sparse_coo_tensor(edge_index, dist_weight * norm)
        side_embed = torch.sparse.mm(dist_adj, x)

        return self.W(side_embed)


class GeoGraph(nn.Module):
    # GeoGraph Neural Network
    def __init__(self, n_user, n_poi, gcn_num, embed_dim, dist_edges, dist_vec, device):
        super(GeoGraph, self).__init__()
        self.n_user = n_user
        self.n_poi = n_poi
        self.embed_dim = embed_dim
        self.gcn_num = gcn_num
        self.device = device

        # construct distance edges tensor
        self.dist_edges = dist_edges.to(device)
        loop_index = torch.arange(0, n_poi).unsqueeze(
            0).repeat(2, 1).to(device)
        self.dist_edges = torch.cat(
            (self.dist_edges, self.dist_edges[[1, 0]], loop_index), dim=-1
        )
        # construct distance vector tensor
        dist_vec = np.concatenate((dist_vec, dist_vec, np.zeros(self.n_poi)))
        self.dist_vec = torch.Tensor(dist_vec).to(device)

        # Projection head module
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # GCN module
        self.gcn = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.gcn.append(Geo_GCN(embed_dim, embed_dim, device).to(device))

        self.init_weights()

        self.selfAttn = SelfAttn(self.embed_dim, 1).to(device)

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

    # split features by batch and compute the mean for each batch
    def split_mean(self, section_feat, sections):
        section_embed = torch.split(section_feat, sections)
        mean_embeds = [torch.mean(embeddings, dim=0)
                       for embeddings in section_embed]
        return torch.stack(mean_embeds)

    def forward(self, data, poi_embeds):
        batch_idx = data.batch
        seq_lens = torch.bincount(batch_idx)
        sections = tuple(seq_lens.cpu().numpy())
        enc = poi_embeds.embeds.weight
        for i in range(self.gcn_num):
            enc = self.gcn[i](enc, self.dist_edges, self.dist_vec)
            enc = F.leaky_relu(enc)
            enc = F.normalize(enc, dim=-1)

        # tar_embed looks like h_t
        tar_embed = enc[data.poi]  # embeddings of the target nodes
        # embeddings of other nodes in the graph
        geo_feat = enc[data.x.squeeze()]

        # add multihead self-attention
        self_attn_feat = self.selfAttn(geo_feat, sections)
        # aggregate self-attention features to obtain semantic representation e_g,u
        aggr_feat = torch.mean(self_attn_feat, dim=1)

        # proj_head(graph_enc) looks like e_g,u, however it does not apply multi-head self-attention
        # return self.proj_head(graph_enc), pred_logits, tar_embed
        return aggr_feat, None, tar_embed
