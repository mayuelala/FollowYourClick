import os
from typing import List

import torch
from torch import nn
from PIL import Image

class TextProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, text_embedding_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        
        self.projection = torch.nn.Linear(text_embedding_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        # nn.init.zeros_(self.projection.weight)
        # nn.init.zeros_(self.projection.bias)
    def forward(self, text_embeds):
        embeds = text_embeds
        embeds = self.projection(embeds)
        embeds = self.norm(embeds)
        return embeds