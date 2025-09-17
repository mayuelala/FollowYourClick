from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward
from .mm_attn_cross import CrossAttention_mm

from einops import rearrange, repeat
import math
from animatediff.models.rope import RoPE

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs,)    
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        video_length                       = 16,
        use_rope_postion_encoding          = False,
        train_video_length                 = 16, 
        add_temporal_lora                  = False,
        rank                               = 4,  
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            use_rope_postion_encoding=use_rope_postion_encoding,
            video_length=video_length,
            train_video_length=train_video_length,
            add_temporal_lora=add_temporal_lora,
            rank=rank
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        use_rope_postion_encoding          = False,
        video_length                       = 16,
        train_video_length                 = 16, 
        add_temporal_lora                  = False,
        rank                               = 4,  
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    use_rope_postion_encoding=use_rope_postion_encoding,
                    video_length=video_length,
                    train_video_length=train_video_length,
                    add_temporal_lora=add_temporal_lora,
                    rank=rank
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        image_length = 0
        video_hidden_states = hidden_states
        image_hidden_states = None
        # if video_length ==1:
        #     return hidden_states
        if encoder_hidden_states.dim()==4:
            video_and_image_num = encoder_hidden_states.shape[1]
            video_num = 1
            image_length = video_and_image_num-video_num
            video_batch_size = encoder_hidden_states.shape[0]
            video_length = hidden_states.shape[2]-image_length
            
            video_encoder_hidden_states = repeat(encoder_hidden_states[:,0,:,:], 'b n c -> b f n c', f=video_length)
            image_encoder_hidden_states = encoder_hidden_states[:,1:,:,:]
            encoder_hidden_states = torch.cat((video_encoder_hidden_states, image_encoder_hidden_states), dim=1)
            encoder_hidden_states = rearrange(encoder_hidden_states, "b f d l -> (b f) d l")
            
            video_hidden_states = hidden_states[:,:,:video_length,:,:]
            image_hidden_states = hidden_states[:,:,video_length:,:,:]            
        else:
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        hidden_states = video_hidden_states
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length, attention_mask=attention_mask)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        if image_hidden_states is not None:
            output = torch.cat((output, image_hidden_states), dim=2)
            
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        use_rope_postion_encoding          = False,
        video_length                       = 16,
        train_video_length                 = 16, 
        add_temporal_lora                  = False,
        rank                               = 4,  
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for block_name in attention_block_types:
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    use_rope_postion_encoding=use_rope_postion_encoding,
                    video_length=video_length,
                    train_video_length=train_video_length,
                    add_temporal_lora=add_temporal_lora,
                    rank=rank
                )
            )
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
                attention_mask=attention_mask
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)
    
class VersatileAttention(CrossAttention_mm):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 24,      
            use_rope_postion_encoding          = False,   
            video_length                       = 16, 
            train_video_length                 = 16,  
            add_temporal_lora                  = False,
            rank                               = 4,  
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        self.use_rope_postion_encoding = use_rope_postion_encoding
        self.pos_encoder = None
        self.add_temporal_lora = add_temporal_lora
        
        if use_rope_postion_encoding:
            self.num_heads = self.heads
            self.rope = RoPE(kwargs["query_dim"]//self.num_heads, temporal_position_encoding_max_len, video_length=video_length, train_video_length=train_video_length)
        else:
            self.pos_encoder = PositionalEncoding(
                kwargs["query_dim"],
                dropout=0., 
                max_len=temporal_position_encoding_max_len
            ) if (temporal_position_encoding and attention_mode == "Temporal") else None
        
        if add_temporal_lora:
            self.rank = rank
            self.to_q_lora = LoRALinearLayer(kwargs["query_dim"], kwargs["query_dim"], rank)
            self.to_k_lora = LoRALinearLayer(kwargs["query_dim"], kwargs["query_dim"], rank)
            self.to_v_lora = LoRALinearLayer(kwargs["query_dim"], kwargs["query_dim"], rank)
            self.to_out_lora = LoRALinearLayer(kwargs["query_dim"], kwargs["query_dim"], rank)
            
    def extra_repr(self):
        return f"(Module Info) Attention_Mode: {self.attention_mode}, Is_Cross_Attention: {self.is_cross_attention}"

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        
        if self.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            if self.pos_encoder is not None and not self.use_rope_postion_encoding:
                hidden_states = self.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if self.add_temporal_lora:
            query = self.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        else:
            query = self.to_q(hidden_states)
        dim = query.shape[-1]
        

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        
        if self.add_temporal_lora:
            key = self.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        else:
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

        if self.use_rope_postion_encoding:
            def split_head(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.num_heads
                tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, head_size, seq_len, dim // head_size)
                return tensor
            def head_to_batch(tensor):
                batch_size, head_size, seq_len, _ = tensor.shape
                head_size = self.num_heads
                return tensor.reshape(batch_size*head_size, seq_len, dim // head_size)
            
            query, key = split_head(query), split_head(key)
            query, key = self.rope(query, key)
            query, key = head_to_batch(query), head_to_batch(key)
        else:
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            
        value = self.reshape_heads_to_batch_dim(value)
        
        # if attention_mask is not None:
        #     if attention_mask.shape[-1] != query.shape[1]:
        #         target_length = query.shape[1]
        #         attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        #         attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        # print(hidden_states.size())
        if attention_mask is not None:   
            attention_mask = rearrange(attention_mask, "b c f h w -> (b h w) f c", f=1)
            attention_mask = 1 - attention_mask.repeat(self.heads, video_length, video_length)
            attention_mask = attention_mask * -10000.0

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        
        # linear proj
        if self.add_temporal_lora:
            hidden_states = self.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        else:
            hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if self.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states

