import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def sequence_mask(lengths, max_len=None):
    lengths_shape = lengths.shape  # torch.size() is a tuple
    lengths = lengths.reshape(-1)
    
    batch_size = lengths.numel()
    max_len = max_len or int(lengths.max())
    lengths_shape += (max_len, )
    
    return (torch.arange(0, max_len, device=lengths.device)
    .type_as(lengths)
    .unsqueeze(0).expand(batch_size,max_len)
    .lt(lengths.unsqueeze(1))).reshape(lengths_shape)


class HardAttn(nn.Module):
    def __init__(self, hidden_size):
        super(HardAttn, self).__init__()
        self.hidden_size = hidden_size
        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

        for w in self.modules():
            if isinstance(w, nn.Linear):
                nn.init.xavier_normal_(w.weight)

    def forward(self, sess_embed, query, sections, seq_lens):
        v_i = torch.split(sess_embed, sections)
        v_i_pad = pad_sequence(v_i, batch_first=True, padding_value=0.)
        
        v_i_pad = self.K(v_i_pad)
        query = self.Q(query)
        seq_mask = sequence_mask(seq_lens)
        
        attn_weight = (v_i_pad * query.unsqueeze(1)).sum(-1)
        pad_val = (-2 ** 32 + 1) * torch.ones_like(attn_weight)
        attn_weight = torch.where(seq_mask, attn_weight, pad_val).softmax(-1)

        seq_feat = (v_i_pad * attn_weight.unsqueeze(-1)).sum(1)
        return self.V(seq_feat)

class GeoGraph(nn.Module):
    def __init__(self, n_user, n_poi, gcn_num, embed_dim, dist_edges, dist_vec, device):
        super(GeoGraph, self).__init__()
        self.n_user = n_user
        self.n_poi = n_poi
        self.embed_dim = embed_dim
        self.gcn_num = gcn_num
        self.device = device

        self.dist_edges = dist_edges.to(device)
        loop_index = torch.arange(0, n_poi).unsqueeze(0).repeat(2, 1).to(device)
        self.dist_edges = torch.cat(
            (self.dist_edges, self.dist_edges[[1, 0]], loop_index), dim=-1
        )
        dist_vec = np.concatenate((dist_vec, dist_vec, np.zeros(self.n_poi)))
        self.dist_vec = torch.Tensor(dist_vec).to(device)

        self.attn = HardAttn(self.embed_dim).to(device)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), 
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

        self.gcn = nn.ModuleList()
        for _ in range(self.gcn_num):
            self.gcn.append(Geo_GCN(embed_dim, embed_dim, device).to(device))

        self.init_weights()

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

    def split_mean(self, section_feat, sections):
        section_embed = torch.split(section_feat, sections)
        mean_embeds = [torch.mean(embeddings, dim=0) for embeddings in section_embed]
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
        tar_embed = enc[data.poi]
        geo_feat = enc[data.x.squeeze()]

        aggr_feat = self.attn(geo_feat, tar_embed, sections, seq_lens)

        graph_enc = self.split_mean(enc[data.x.squeeze()], sections)
        pred_input = torch.cat((aggr_feat, tar_embed), dim=-1)

        pred_logits = self.predictor(pred_input)

        # proj_head(graph_enc) looks like e_g,u, however it does not apply multi-head self-attention
        return self.proj_head(graph_enc), pred_logits, tar_embed
    
class Geo_GCN(nn.Module):
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
