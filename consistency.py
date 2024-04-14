import torch
from torch import nn
import math
import torch.nn.functional as F


class SampleSimilarities(nn.Module):
    def __init__(self, embed_dim, queue_size, T, device):
        """
        Calculate cosine similarity for each batch.

        Args:
            embed_dim (int): The dimension of the embedding of semantic representation.
            queue_size (int): The size of the memory queue.
            T (float): The softmax temperature for the similarity calculation.
            device (torch.device): The device to be used for computation.
        """
        super(SampleSimilarities, self).__init__()
        self.inputSize = embed_dim
        self.queueSize = queue_size
        self.T = T
        self.index = 0
        self.device = device

        # initialize the memory bank (a uniform distribution between -stdv and stdv)
        stdv = 1. / math.sqrt(embed_dim / 3)
        initial_memory = torch.rand(queue_size, embed_dim) * 2 * stdv - stdv
        self.register_buffer('memory',initial_memory)

    def forward(self, input_embed, update=True):
        """
        Compute the similarities between each sequence and all anchor sequences.

        Args:
            input_embed (torch.Tensor): The input sequence.
            update (bool): Whether to update the anchor sequences.

        Returns:
            torch.Tensor: The computed similarities.
        """
        batchSize = input_embed.shape[0]
        anchorSeq = self.memory.clone()

        # Compute the cosine similarity
        input_n, anchor_n = input_embed.norm(dim=1)[:,None], anchorSeq.norm(dim=1)[:,None]
        norm_input = input_embed / torch.max(input_n, torch.tensor([1e-6]).to(self.device))
        norm_anchor = anchorSeq / torch.max(anchor_n, torch.tensor([1e-6]).to(self.device))
        cosSim = torch.mm(norm_input, norm_anchor.t())

        # Scale by temperature
        cosSim = torch.div(cosSim, self.T)

        # Compute the softmax
        # sim = F.log_softmax(cosSim, dim=1) # log_softmax instead of softmax: for numerical stability
        cosSim = cosSim.contiguous()

        # Update the memory bank
        if update:
            with torch.no_grad():
                # Compute the indices for updating anchor sequence in the memory bank
                new_idx = torch.arange(batchSize).to(self.device)
                new_idx += self.index
                new_idx = torch.fmod(new_idx, self.queueSize)
                new_idx = new_idx.long()

                # Update the memory bank with the input sequence
                self.memory.index_copy_(dim=0, index=new_idx, source=input_embed)

                # Update the index for the next update
                self.index = (self.index + batchSize) % self.queueSize

        return cosSim


class consistencyLoss(nn.Module):
    """
    Calculates the consistency loss between two sets of embeddings.

    Args:
        embed_dim (int): The dimension of the embeddings.
        queue_size (int): The size of the queue used for calculating sample similarities.
        T (float): The temperature parameter for calculating sample similarities.
        device (str): The device on which the calculations will be performed.

    Attributes:
        calculate_sampleSimilarities (SampleSimilarities): An instance of the SampleSimilarities class used for calculating sample similarities.

    Methods:
        forward(seq_embed, geo_embed): Calculates the consistency loss between the given sequence and geometry embeddings.

    """

    def __init__(self, embed_dim, queue_size, T, device):
        super(consistencyLoss, self).__init__()
        self.calculate_sampleSimilarities = SampleSimilarities(embed_dim, queue_size, T, device).to(device)

    def forward(self, seq_embed, geo_embed):
            """
            Calculates the consistency loss between the given sequence and geometry embeddings.

            Args:
                seq_embed (torch.Tensor): The sequence embeddings.
                geo_embed (torch.Tensor): The geometry embeddings.

            Returns:
                torch.Tensor: The consistency loss.

            Raises:
                None

            Examples:
                >>> model = ConsistencyModel()
                >>> seq_embed = torch.randn(32, 128)
                >>> geo_embed = torch.randn(32, 128)
                >>> loss = model.forward(seq_embed, geo_embed)
            """
            # Calculate the sample similarities for the sequence and geometry embeddings
            seq_simDistribution = self.calculate_sampleSimilarities(seq_embed)
            geo_simDistribution = self.calculate_sampleSimilarities(geo_embed)
            
            # Apply the softmax function to the distributions
            seq_simDistribution = F.log_softmax(seq_simDistribution, dim=1)
            geo_simDistribution = F.softmax(geo_simDistribution, dim=1)
            
            # Calculate the KL divergence between the two distributions
            conLoss = F.kl_div(seq_simDistribution,geo_simDistribution,reduction='batchmean')

            return conLoss
