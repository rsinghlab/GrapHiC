import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import TransformerConv
import torch.nn as nn
from torch_geometric.nn.norm import GraphNorm



class GPSConv(torch.nn.Module):
    """The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper.

    The GPS layer is based on a 3-part recipe:

    1. Inclusion of positional (PE) and structural encodings (SE) to the input
       features (done in a pre-processing step via
       :class:`torch_geometric.transforms`).
    2. A local message passing layer (MPNN) that operates on the input graph.
    3. A global attention layer that operates on the entire graph.
    """
    
    def __init__(
        self,
        channels: int,
        edge_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act=torch.relu
    ):
        super().__init__()

        self.channels = channels
        self.act = act
        self.heads = heads
        self.dropout = dropout
        
        self.conv = TransformerConv(
            channels,
            channels,
            heads=heads,
            edge_dim=edge_dim,
            beta=True,
            dropout=0.3
        )
        self.linear = Linear(
            channels*heads, 
            channels
        ) 

        self.attn = torch.nn.MultiheadAttention(
            channels,
            heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            act,
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )
        self.bn = GraphNorm(channels)
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor 
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
    
        h = self.conv(x, edge_index, edge_attr)
        h = self.linear(h)
        
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        h = self.bn(h)
        hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        h = self.bn(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        out = self.bn(out)
        return out