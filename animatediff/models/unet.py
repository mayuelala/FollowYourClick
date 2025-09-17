# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import os
import json
import pdb

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
    get_down_block,
    get_up_block,
)
from .resnet import InflatedConv3d, PseudoConv3d, InflatedGroupNorm
from .image_adapter import ImageProjModel
from .condition_module import TextProjModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,      
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        
        # Additional
        use_motion_module              = False,
        motion_module_resolutions      = ( 1,2,4,8 ),
        motion_module_mid_block        = False,
        motion_module_decoder_only     = False,
        motion_module_type             = None,
        motion_module_kwargs           = {},
        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention    = None,
        use_pseudo_conv3d              = False,
        use_first_frame_condition_concat = False,
        image_condition_dim            = 1024,
        use_ip_cross_attention         = False,
        scale                          = 1.0,
        num_tokens                     = 4,
        use_camera_motion_condition    = False,
        use_text_encoder_2             = False,
        text_encoder_2_dim             = 4096,
        use_inflated_groupnorm         = False,
        use_fps_condition              = False,
        use_temporal_conv              = False,
        use_first_frame_mask_condition_concat = False,
        
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        # assert (use_first_frame_condition_concat and use_first_frame_mask_condition_concat) == False

        if use_pseudo_conv3d:
            if use_first_frame_condition_concat:
                self.conv_in = PseudoConv3d(in_channels*2, block_out_channels[0], kernel_size=3, padding=(1, 1))
            elif use_first_frame_mask_condition_concat:
                self.conv_in = PseudoConv3d(in_channels*2+1, block_out_channels[0], kernel_size=3, padding=(1, 1))
            else:
                self.conv_in = PseudoConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        else:
            if use_first_frame_condition_concat:
                self.conv_in = InflatedConv3d(in_channels*2, block_out_channels[0], kernel_size=3, padding=(1, 1))
            elif use_first_frame_mask_condition_concat:
                self.conv_in = InflatedConv3d(in_channels*2+1, block_out_channels[0], kernel_size=3, padding=(1, 1))
            else:
                self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        
        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        if use_camera_motion_condition:
            self.camera_motion_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            nn.init.zeros_(self.camera_motion_embedding.linear_2.weight)
            nn.init.zeros_(self.camera_motion_embedding.linear_2.bias)
        
        if use_fps_condition:
            self.fps_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            nn.init.zeros_(self.fps_embedding.linear_2.weight)
            nn.init.zeros_(self.fps_embedding.linear_2.bias)

            self.motion_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            nn.init.zeros_(self.motion_embedding.linear_2.weight)
            nn.init.zeros_(self.motion_embedding.linear_2.bias)
            
        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        if use_ip_cross_attention:
            self.image_proj_model = None
            # self.image_proj_model = ImageProjModel(
            #     cross_attention_dim=cross_attention_dim,
            #     clip_embeddings_dim=image_condition_dim,
            #     clip_extra_context_tokens=4,
            # )
        
            # image_proj_model = Resampler(
            #     dim=self.pipe.unet.config.cross_attention_dim,
            #     depth=4,
            #     dim_head=64,
            #     heads=12,
            #     num_queries=self.num_tokens,
            #     embedding_dim=self.image_encoder.config.hidden_size,
            #     output_dim=self.pipe.unet.config.cross_attention_dim,
            #     ff_mult=4,
            # )
            
        if use_text_encoder_2:
            self.text_encoder_proj_model_t5 = TextProjModel(
                text_embedding_dim=text_encoder_2_dim,
                cross_attention_dim=cross_attention_dim
            )
            
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_pseudo_conv3d=use_pseudo_conv3d,
                
                use_motion_module=use_motion_module and (res in motion_module_resolutions) and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                
                use_ip_cross_attention = use_ip_cross_attention,
                scale=scale,
                num_tokens=num_tokens,
                
                use_text_encoder_2=use_text_encoder_2,
                
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_temporal_conv=use_temporal_conv
                
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_pseudo_conv3d=use_pseudo_conv3d,
                
                
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                
                use_ip_cross_attention = use_ip_cross_attention,
                scale=scale,
                num_tokens=num_tokens,
                use_text_encoder_2=use_text_encoder_2,
                
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_temporal_conv=use_temporal_conv
                
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_pseudo_conv3d=use_pseudo_conv3d,
                

                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                
                use_ip_cross_attention = use_ip_cross_attention,
                scale=scale,
                num_tokens=num_tokens,
                use_text_encoder_2=use_text_encoder_2,
                
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_temporal_conv=use_temporal_conv
                
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
            
        self.conv_act = nn.SiLU()
        if use_pseudo_conv3d:
            self.conv_out = PseudoConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        else:
            self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
    
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_first_frame_condition: bool=False,
        use_first_frame_condition_concat: bool=False,
        use_ip_cross_attention: bool=False,
        reference_images_latent=None,
        reference_images_clip_feat=None,
        use_camera_motion_condition=False,
        camera_movement_type_tensor=None,
        use_image_concat_training=False,
        use_text_encoder_2=False,
        encoder_hidden_states_2=None,
        use_fps_condition=False,
        fps_tensor=None,
        first_images_mask=None,
        flow_control=None
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # e.g., sample.shape: torch.Size([4, 4, 16, 32, 56]) # batch, channel, height, width
        # e.g., timestep.shape: torch.Size([4])
        # e.g., encoder_hidden_states.shape: torch.Size([4, 77, 768]) # from text encoder

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
            
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        if use_camera_motion_condition==True:
            if not torch.is_tensor(camera_movement_type_tensor):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                camera_movement_type_tensor = torch.tensor([camera_movement_type_tensor], dtype=dtype, device=sample.device)
            elif len(camera_movement_type_tensor.shape) == 0:
                camera_movement_type_tensor = camera_movement_type_tensor[None].to(sample.device)
        
        if use_fps_condition:
            if not torch.is_tensor(fps_tensor):
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                fps_tensor = torch.tensor([fps_tensor], dtype=dtype, device=sample.device)
            elif len(fps_tensor.shape) == 0:
                fps_tensor = fps_tensor[None].to(sample.device)
            
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        if use_first_frame_condition:
            timesteps = torch.cat((timesteps, torch.zeros(1).to(timesteps.device).to(timesteps.dtype)))

        t_emb = self.time_proj(timesteps)
        # e.g., timesteps.shape: torch.Size([4]);
        # e.g., t_emb.shape: torch.Size([4, 320])
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        # e.g., emb.shape: torch.Size([4, 1280])
        # initially, emb meaning time embedding

        if use_camera_motion_condition==True:
            camera_movement_types = camera_movement_type_tensor.expand(sample.shape[0]).to(t_emb.device)
            camera_movement_types_emb = self.time_proj(camera_movement_types)
            
            camera_movement_types_emb = camera_movement_types_emb.to(dtype=self.dtype)
            emb += self.camera_motion_embedding(camera_movement_types_emb)
            # can add camera_motion_embedding to emb
            
        if use_fps_condition==True:
            #================================================
            fps_tensor = fps_tensor.expand(sample.shape[0]).to(t_emb.device)
            fps_emb = self.time_proj(fps_tensor)
            
            fps_emb = fps_emb.to(dtype=self.dtype)
            emb += self.fps_embedding(fps_emb)
            # can add fps_embedding to emb
            
            flow_control = flow_control.expand(sample.shape[0]).to(t_emb.device)
            flow_emb = self.time_proj(flow_control)
            
            flow_emb = flow_emb.to(dtype=self.dtype)
            emb += self.motion_embedding(flow_emb)
            #================================================
            # flow_control = flow_control.expand(sample.shape[0]).to(t_emb.device)
            # flow_emb = self.time_proj(flow_control)
            
            # flow_emb = flow_emb.to(dtype=self.dtype)
            # emb += self.fps_embedding(flow_emb)
        
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
            # can add class_emb to emb
        
        num_frames = sample.shape[2]
        if use_first_frame_condition_concat and reference_images_latent is not None:
            sample_first = reference_images_latent
            # sample_first = reference_images_latent if reference_images_latent is not None else sample[:,:,0,:,:]
            sample_first = sample_first.unsqueeze(2).repeat(1,1,num_frames,1,1)
            # sample_first = torch.cat((sample_first.unsqueeze(2), torch.zeros_like(sample_first.unsqueeze(2)).repeat(1,1,num_frames-1,1,1)), dim=2)
            sample = torch.cat((sample, sample_first), dim=1)
            # sample = sample
        # pre-process
        sample = self.conv_in(sample)
        # e.g. sample shape becomes: torch.Size([4, 4, 16, 32, 56]) --> torch.Size([4, 320, 16, 32, 56])
        
        if use_first_frame_condition_concat:
            sample = sample/2

        if use_ip_cross_attention:
            ip_tokens = self.image_proj_model(reference_images_clip_feat)
            encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
            

        if use_text_encoder_2:
            encoder_hidden_states_2_proj = self.text_encoder_proj_model_t5(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2_proj], dim=0)
        
        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    # attention_mask=attention_mask,
                    first_images_mask=first_images_mask
                )
                # e.g., sample shape becomes:
                # torch.Size([4, 320, 16, 32, 56])
                # --> torch.Size([4, 320, 16, 16, 28])
                # --> torch.Size([4, 640, 16, 8, 14])
                # --> torch.Size([4, 1280, 16, 4, 7])

                # e.g., res_samples is tuple, after the 1st downsample_block
                # res_samples[0].shape: torch.Size([4, 320, 16, 32, 56])
                # res_samples[1].shape: torch.Size([4, 320, 16, 32, 56])
                # res_samples[2].shape: torch.Size([4, 320, 16, 16, 28])

            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states)

            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
            first_images_mask=first_images_mask
        )
        # after mid, sample.shape: torch.Size([4, 1280, 16, 4, 7])

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    first_images_mask=first_images_mask
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size, encoder_hidden_states=encoder_hidden_states,
                )
        
        # e.g., now sample.shape: torch.Size([4, 320, 16, 32, 56])

        # post-process
        sample = self.conv_norm_out(sample) #e.g., sample.shape is still torch.Size([4, 320, 16, 32, 56])
        sample = self.conv_act(sample) #e.g., sample.shape is still torch.Size([4, 320, 16, 32, 56])
        sample = self.conv_out(sample) #e.g., sample.shape becomes torch.Size([4, 4, 16, 32, 56])

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
        # import pdb; pdb.set_trace()
        
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded temporal unet's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config, **unet_additional_kwargs)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        
        state_dict = torch.load(model_file, map_location="cpu")
        
        if ('use_first_frame_condition_concat' in unet_additional_kwargs and unet_additional_kwargs['use_first_frame_condition_concat']) or \
            ('use_first_frame_mask_condition_concat' in unet_additional_kwargs and unet_additional_kwargs['use_first_frame_mask_condition_concat']): # means we load the pretrained 3d unet model and we need to extend the conv_in by hand
            model.conv_in.weight.data.zero_()
            # import pdb; pdb.set_trace()
            model.state_dict()['conv_in.weight'][:,:4,:,:] = state_dict['conv_in.weight']
            # model.state_dict()['conv_in.weight'] = torch.cat([state_dict['conv_in.weight']] * 2, dim=1)
            for k, v in model.state_dict().items():
                if 'conv_in' in k:
                    state_dict.update({k: v})

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        # print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n")
        
        params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
        print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")
        
        return model
