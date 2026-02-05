import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
import lib.utils as utils
from layers.basic_att import BasicAtt

class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2, geo_feats=None):
        #print("att_map shape before geo_feats:", att_map.shape)

        if self.attention_basic is not None:
            #print("att_map shape for basic attention:", att_map.shape)
            att_map = self.attention_basic(att_map)
            #print("att_map shape After basic attention:", att_map.shape)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)
        #print("att_map_pool shape:", att_map_pool.shape)

        alpha_spatial = self.attention_last(att_map)
        #print("alpha_spatial shape after attention_last:", alpha_spatial.shape)

        alpha_channel = self.attention_last2(att_map_pool)
        #print("alpha_channel shape after attention_last2:", alpha_channel.shape)

        alpha_channel = torch.sigmoid(alpha_channel)
        #print("alpha_channel shape after sigmoid:", alpha_channel.shape)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, -1e9)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)
        #print("alpha_spatial shape after softmax:", alpha_spatial.shape)

        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)
        #print("value2 shape after matmul:", value2.shape)

        attn = value1 * value2 * alpha_channel
        #print("attn shape:", attn.shape)
        return attn