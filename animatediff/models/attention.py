# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat
import pdb
import math

class IPCrossAttention(CrossAttention):
    def __init__(self, query_dim: int,
                cross_attention_dim: Optional[int] = None,
                heads: int = 8,
                dim_head: int = 64,
                dropout: float = 0.0,
                bias=False,
                upcast_attention: bool = False,
                upcast_softmax: bool = False,
                added_kv_proj_dim: Optional[int] = None,
                norm_num_groups: Optional[int] = None, scale=1.0, num_tokens=4):
        super().__init__(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention,
        )
        
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or query_dim, query_dim, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or query_dim, query_dim, bias=False)
        
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            
        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

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
        
        
        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = self.reshape_heads_to_batch_dim(ip_key)
        ip_value = self.reshape_heads_to_batch_dim(ip_value)
        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            ip_hidden_states = self._memory_efficient_attention_xformers(query, ip_key, ip_value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            ip_hidden_states = ip_hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                ip_hidden_states = self._attention(query, ip_key, ip_value, attention_mask)
            else:
                ip_hidden_states = self._sliced_attention(query, ip_key, ip_value, sequence_length, dim, attention_mask)
                
            
        hidden_states = hidden_states + self.scale * ip_hidden_states
        
        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        
        use_ip_cross_attention = False,
        scale=1.0,
        num_tokens=4,
        
        use_text_encoder_2=False
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.use_text_encoder_2 = use_text_encoder_2
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,

                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                    
                    use_ip_cross_attention = use_ip_cross_attention,
                    scale=scale,
                    num_tokens=num_tokens,
                    use_text_encoder_2=use_text_encoder_2
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        image_length = 0
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states_2 = None
        if encoder_hidden_states.dim()==4:
            # import pdb; pdb.set_trace()
            video_and_image_num = encoder_hidden_states.shape[1]
            video_num = 1
            image_length = video_and_image_num-video_num
            if self.use_text_encoder_2:
                real_batch_size = encoder_hidden_states.shape[0]//2
                encoder_hidden_states_1 =encoder_hidden_states[:real_batch_size]
                encoder_hidden_states_2 =encoder_hidden_states[real_batch_size:]
                encoder_hidden_states = encoder_hidden_states_1
                
                video_batch_size = encoder_hidden_states.shape[0]
                video_length = hidden_states.shape[0]//video_batch_size-image_length
                video_encoder_hidden_states = repeat(encoder_hidden_states[:,0,:,:], 'b n c -> b f n c', f=video_length)
                image_encoder_hidden_states = encoder_hidden_states[:,1:,:,:]
                encoder_hidden_states = torch.cat((video_encoder_hidden_states, image_encoder_hidden_states), dim=1)
                encoder_hidden_states = rearrange(encoder_hidden_states, "b f d l -> (b f) d l")
                
                video_encoder_hidden_states_2 = repeat(encoder_hidden_states_2[:,0,:,:], 'b n c -> b f n c', f=video_length)
                image_encoder_hidden_states_2 = encoder_hidden_states_2[:,1:,:,:]
                encoder_hidden_states_2 = torch.cat((video_encoder_hidden_states_2, image_encoder_hidden_states_2), dim=1)
                encoder_hidden_states_2 = rearrange(encoder_hidden_states_2, "b f d l -> (b f) d l")
                
            else:
                video_batch_size = encoder_hidden_states.shape[0]
                video_length = hidden_states.shape[0]//video_batch_size-image_length
                video_encoder_hidden_states = repeat(encoder_hidden_states[:,0,:,:], 'b n c -> b f n c', f=video_length)
                image_encoder_hidden_states = encoder_hidden_states[:,1:,:,:]
                encoder_hidden_states = torch.cat((video_encoder_hidden_states, image_encoder_hidden_states), dim=1)
                encoder_hidden_states = rearrange(encoder_hidden_states, "b f d l -> (b f) d l")
        else:
            if self.use_text_encoder_2:
                real_batch_size = encoder_hidden_states.shape[0]//2
                encoder_hidden_states_1 =encoder_hidden_states[:real_batch_size]
                encoder_hidden_states_2 =encoder_hidden_states[real_batch_size:]
                encoder_hidden_states_1 = repeat(encoder_hidden_states_1, 'b n c -> (b f) n c', f=video_length)
                encoder_hidden_states_2 = repeat(encoder_hidden_states_2, 'b n c -> (b f) n c', f=video_length)
                encoder_hidden_states = encoder_hidden_states_1
            else:
                
                encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                image_length=image_length,
                encoder_hidden_states_2=encoder_hidden_states_2
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=(video_length+image_length))
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

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
    
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention = None,
        
        use_ip_cross_attention = False,
        scale=1.0,
        num_tokens=4,
        
        use_text_encoder_2=False
        
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.use_ip_cross_attention = use_ip_cross_attention
        self.use_text_encoder_2=use_text_encoder_2
        
        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            self.attn1 = SparseCausalAttention2D(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            if use_ip_cross_attention:
                self.attn2 = IPCrossAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    scale=scale, num_tokens=num_tokens
                )
            else:
                self.attn2 = CrossAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.pos_encoder = PositionalEncoding(
                dim,
                dropout=0., 
                max_len=32
            )
            self.attn_temp = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        if use_text_encoder_2:
            if cross_attention_dim is not None:
                self.attn_t5 = CrossAttention(
                        query_dim=dim,
                        cross_attention_dim=cross_attention_dim,
                        heads=num_attention_heads,
                        dim_head=attention_head_dim,
                        dropout=dropout,
                        bias=attention_bias,
                        upcast_attention=upcast_attention,
                    )
                nn.init.zeros_(self.attn_t5.to_out[0].weight.data)
                nn.init.zeros_(self.attn_t5.to_out[0].bias.data)
            else:
                self.attn_t5 = None
            
            if cross_attention_dim is not None:
                self.norm_t5 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
            else:
                self.norm_t5 = None
            
        
    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, image_length=0, encoder_hidden_states_2=None):
        # SparseCausal-Attention
        # import pdb; pdb.set_trace()
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        # if self.only_cross_attention:
        #     hidden_states = (
        #         self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        #     )
        # else:
        #     hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # pdb.set_trace()
        if self.unet_use_cross_frame_attention:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            # print (f'{norm_hidden_states.shape}-{encoder_hidden_states.shape}-{encoder_hidden_states_2.shape}')
            # print (f'{norm_hidden_states.shape}-{encoder_hidden_states.shape}')
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # print (f'{type(self).__name__}-use_text_encoder_2-{self.use_text_encoder_2}')
        if self.use_text_encoder_2 and self.attn_t5 is not None:
            # print (f'{type(self).__name__}-use_text_encoder_2-{self.use_text_encoder_2}--{encoder_hidden_states_2.shape}')
            
            # import pdb; pdb.set_trace()
            norm_hidden_states = (
                self.norm_t5(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_t5(hidden_states)
            )
            hidden_states = (
                self.attn_t5(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states_2, attention_mask=attention_mask
                )
                + hidden_states
            )

            
        # Temporal-Attention
        if self.unet_use_temporal_attention:
            total_length = video_length+image_length
            batch_size = hidden_states.shape[0] // total_length
            hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", b=batch_size)
            video_hidden_states = hidden_states[:,:video_length, :, :]
            video_hidden_states = rearrange(video_hidden_states, "b f d c -> (b f) d c")
            image_hidden_states = None
            if image_length>0: image_hidden_states = hidden_states[:,video_length:, :, :]
            d = video_hidden_states.shape[1]
            video_hidden_states = rearrange(video_hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            # video_hidden_states = self.pos_encoder(video_hidden_states)
            norm_hidden_states = (
                self.norm_temp(self.pos_encoder(video_hidden_states), timestep) if self.use_ada_layer_norm else self.norm_temp(video_hidden_states)
            )
            video_hidden_states = self.attn_temp(norm_hidden_states) + video_hidden_states
            video_hidden_states = rearrange(video_hidden_states, "(b d) f c -> b f d c", d=d)
            
            if image_hidden_states is not None: hidden_states = torch.cat((video_hidden_states, image_hidden_states), dim=1)
            else: hidden_states = video_hidden_states
            hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c")
            
            
        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states
