import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout_rate=0.1):
        super(GeoAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear layers to project the input features for multi-head attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, att_feats, geo_feats):
        batch_size = att_feats.size(0)

        # Project the inputs to multiple heads
        queries = self.query_proj(att_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(geo_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(geo_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to the values
        attention_output = torch.matmul(attention_weights, values)

        # Concatenate the outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Apply dropout before final linear transformation
        attention_output = self.dropout(self.out_proj(attention_output))

        # Combine with the original att_feats using residual connection and layer normalization
        output = self.layer_norm(att_feats + geo_feats * attention_output)

        return output
