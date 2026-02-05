import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
import layers


class LowRank(nn.Module):
    def __init__(self, embed_dim, geo_dim, att_type, att_heads, att_mid_dim, att_mid_drop):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.geo_dim = geo_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = 2 * embed_dim if cfg.MODEL.BILINEAR.ACT == 'GLU' else embed_dim

        # Initialize input projections for queries, keys, and values
        self.in_proj_q = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            torch.nn.GroupNorm(self.num_heads, embed_dim)
        )

        self.in_proj_k = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            torch.nn.GroupNorm(self.num_heads, embed_dim)
        )

        self.in_proj_v1 = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            torch.nn.GroupNorm(self.num_heads, embed_dim)
        )

        self.in_proj_v2 = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            utils.activation(cfg.MODEL.BILINEAR.ACT),
            torch.nn.GroupNorm(self.num_heads, embed_dim)
        )

        # Attention network
        self.attn_net = layers.create(att_type, att_mid_dim, att_mid_drop)

        # Buffering states for fast forward computation
        self.clear_buffer()

    def apply_to_states(self, fn):
        """Apply a function to internal states"""
        self.buffer_keys = fn(self.buffer_keys)
        self.buffer_value2 = fn(self.buffer_value2)

    def init_buffer(self, batch_size):
        """Initialize buffer with batch size and head dimension"""
        self.buffer_keys = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()
        self.buffer_value2 = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()

    def clear_buffer(self):
        """Clear internal buffers"""
        self.buffer_keys = None
        self.buffer_value2 = None

    def forward(self, query, key, mask, value1, value2, geo_feats=None, precompute=False):
        batch_size = query.size(0)

        # Project inputs to their respective dimensions
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        # Reshape projections to (batch_size, num_heads, head_dim)
        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if not precompute:
            key = key.view(-1, key.size(-1))
            value2 = value2.view(-1, value2.size(-1))
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)

            # Reshape and transpose key and value tensors
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2

        # Compute scaled dot-product attention
        attn_map = torch.matmul(q.unsqueeze(-2), k.transpose(-2, -1)) * self.scaling

        # Optionally repeat dimensions
        attn_map = attn_map.transpose(-2, -1).repeat(1, 1, 1, self.head_dim)

        # Apply attention network
        attn = self.attn_net(attn_map, mask, v1, v2, geo_feats)

        # Reshape the attention output
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2, geo_feats=None):
        """Precompute key and value projections for future usage"""
        batch_size = value2.size(0)
        key = key.view(-1, key.size(-1))
        value2 = value2.view(-1, value2.size(-1))

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2
