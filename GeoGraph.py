import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import degree


class SelfAttn(nn.Module):
    """
    Self-Attention module that applies multi-head attention mechanism on input embeddings.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        n_heads (int): The number of attention heads to use.

    Attributes:
        multihead_attn (nn.MultiheadAttention): The multi-head attention module.

    """

    def __init__(self, embed_dim, n_heads):
        super(SelfAttn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, sess_embed, sections):
        """
        Forward pass of the SelfAttn module.

        Args:
            sess_embed (torch.Tensor): The input session embeddings.
            sections (List[int]): A list of section lengths for each session.

        Returns:
            torch.Tensor: The output attention embeddings.

        """
        v_i = torch.split(sess_embed, sections)
        v_i_pad = pad_sequence(v_i, batch_first=True, padding_value=0.)

        attn_output, _ = self.multihead_attn(v_i_pad, v_i_pad, v_i_pad)

        return attn_output


class GraphLayer(nn.Module):
    """
    A single message passing nueral network layer

    Args:
        embed_dim (int): Dimension of the embeddings.

    """

    def __init__(self, embed_dim):
        super(GraphLayer, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, poi_rep, edge_index, dist_vec):
        """
        Args:
            poi_rep (torch.Tensor): Input poi representation (num_pois, embed_dim).
            edge_index (torch.Tensor): Edge index (2, num_edges).
            dist_vec (torch.Tensor): Distance vector (num_edges,).

        Returns:
            torch.Tensor: Updated poi representation (num_pois, embed_dim).
        """
        
        nodes1, nodes2 = edge_index
        node_degree = degree(nodes1, poi_rep.size(0), dtype=poi_rep.dtype)
        norm_weight = torch.pow(node_degree[nodes1] * node_degree[nodes2], -0.5)
        dist_weight = torch.exp(-(dist_vec ** 2))
        
        weight_mat = torch.sparse_coo_tensor(edge_index, norm_weight * dist_weight)

        poi_rep = self.linear(poi_rep)
        poi_rep = torch.sparse.mm(weight_mat, poi_rep)
        poi_rep = F.leaky_relu(poi_rep)
        return F.normalize(poi_rep, dim=-1)


class GeoGraph(nn.Module):
    """
    Geographical model.

    Args:
        n_poi (int): Number of points of interest (POIs) in the graph.
        n_layers (int): Number of message passing nueral network layers.
        embed_dim (int): Dimension of the node embeddings.
        dist_edges (torch.Tensor): Tensor representing the edges in the graph, size (2, num_edges).
        dist_vec (np.ndarray): Array representing the distance of the edges, size (num_edges,).
        n_heads (int): Number of attention heads in the self-attention mechanism.

    """

    def __init__(self, n_poi, n_layers, embed_dim, dist_edges, dist_vec, n_heads):
        super(GeoGraph, self).__init__()
        self.n_layers = n_layers
        
        # add the reverse direction and self-loop to the distance edges
        self.dist_edges = dist_edges
        loop_index = torch.arange(0, n_poi).unsqueeze(0).repeat(2, 1)
        self.dist_edges = torch.cat((self.dist_edges, self.dist_edges[[1, 0]], loop_index), dim=-1)
        
        # add the reverse direction and self-loop to the distance vector
        dist_vec = np.concatenate((dist_vec, dist_vec, np.zeros(n_poi)))
        self.dist_vec = torch.Tensor(dist_vec)

        # message passing neural network layers
        self.mpnn = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mpnn.append(GraphLayer(embed_dim))

        # self-attention layer
        self.selfAttn = SelfAttn(embed_dim, n_heads)
        
        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights in the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, data, poi_embeds):
        """
        Forward pass of the model.

        Args:
            data: Input data.
            poi_embeds: Embeddings of the points of interest (POIs).

        Returns:
            aggr_feat: Aggregated features obtained from self-attention mechanism.
            tar_embed: Embeddings of the target nodes.
        """
        # the original embeddings of the POIs
        enc = poi_embeds.embeds.weight
        # apply GCN layers
        for i in range(self.n_layers):
            enc = self.mpnn[i](enc, self.dist_edges, self.dist_vec)
        
        # geographical encoding for target poi
        tar_embed = enc[data.poi]
        
        # get sequence lengths
        _, seq_len = torch.unique(data.batch, return_counts=True)
        sections = tuple(seq_len.cpu().numpy())
        
        # apply multihead self-attention
        poi_embed_in_seq = enc[data.x.squeeze()] # embeddings for poi in the sequence
        self_attn_feat = self.selfAttn(poi_embed_in_seq, sections)
        # aggregate self-attention features to obtain semantic representation e_g,u
        aggr_feat = torch.mean(self_attn_feat, dim=1)

        return aggr_feat, tar_embed
