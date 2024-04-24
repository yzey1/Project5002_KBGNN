import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    '''Embedding layer for POI. 

    Args:
        n_poi (int): Number of POI.
        embed_dim (int): Embedding dimension.

    Input:
        torch.Tensor: Index of POI.

    Output:
        torch.Tensor: Embedding vector of POI.
    '''

    def __init__(self, n_poi, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeds = nn.Embedding(n_poi, embed_dim)
        nn.init.xavier_normal_(self.embeds.weight)

    def forward(self, idx):
        return self.embeds(idx)


class MLP(nn.Module):
    '''MLP for predicting the probability of visiting a POI.

    Args:
        embed_dim (int): Embedding dimension.

    Input:
        torch.Tensor: Geographical embedding.
        torch.Tensor: Sequential embedding.
        torch.Tensor: Target geographical representation.

    Output:
        torch.Tensor: Logits of the prediction.
    '''

    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        self.embed_dim = embed_dim

        self.predictor = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, e_g, e_s, h_t):
        flat_input = torch.cat((e_g, e_s, h_t), dim=-1)
        pred_logits = self.predictor(flat_input)

        return pred_logits


class MLP2(nn.Module):
    '''MLP for predicting the probability of visiting a POI.
       only use one embedding and target geographical representation.

    Args:
        embed_dim (int): Embedding dimension.

    Input:
        torch.Tensor: one embedding.
        torch.Tensor: Target geographical representation.

    Output:
        torch.Tensor: Logits of the prediction.
    '''

    def __init__(self, embed_dim):
        super(MLP2, self).__init__()
        self.embed_dim = embed_dim

        self.predictor = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, e_g, h_t):
        flat_input = torch.cat((e_g, h_t), dim=-1)
        pred_logits = self.predictor(flat_input)

        return pred_logits
