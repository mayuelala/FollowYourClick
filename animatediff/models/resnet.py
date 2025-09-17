# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class InflatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

class TemporalConvBlock(nn.Module):
    """
    Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    """

    def __init__(self, in_channels, out_channels=None, dropout=0.0, spatial_aware=False):
        super(TemporalConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_shape = (3, 1, 1) if not spatial_aware else (3, 3, 3)
        padding_shape = (1, 0, 0) if not spatial_aware else (1, 1, 1)

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels), nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_shape, padding=padding_shape))
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, kernel_shape, padding=padding_shape))
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, (3, 1, 1), padding=(1, 0, 0)))
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv3d(out_channels, in_channels, (3, 1, 1), padding=(1, 0, 0)))

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x + identity
    
class PseudoConv3d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, temporal_kernel_size=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
        if temporal_kernel_size is None:
            temporal_kernel_size = kernel_size

        self.conv_temporal = (
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_kernel_size // 2,
            )
            if kernel_size > 1
            else None
        )

        if self.conv_temporal is not None:
            nn.init.dirac_(self.conv_temporal.weight.data)  # initialized to be identity
            nn.init.zeros_(self.conv_temporal.bias.data)

    def forward(self, x):
        b = x.shape[0]

        is_video = x.ndim == 5
        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")

        x = super().forward(x)  

        if is_video:
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b)

        if self.conv_temporal is None or not is_video:
            return x

        *_, h, w = x.shape

        x = rearrange(x, "b c f h w -> (b h w) c f")

        x = self.conv_temporal(x)   # 加入空间1D的时序卷积。channel不变。（建模时序信息）

        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)

        return x
    

class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # if self.use_conv:
        #     if self.name == "conv":
        #         hidden_states = self.conv(hidden_states)
        #     else:
        #         hidden_states = self.Conv2d_0(hidden_states)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        use_pseudo_conv3d=False,
        use_inflated_groupnorm=None,
        use_temporal_conv=False
        
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor
        self.use_temporal_conv = use_temporal_conv
        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        assert use_inflated_groupnorm != None
        if use_inflated_groupnorm:
            self.norm1 = InflatedGroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
            
        if use_pseudo_conv3d:
            self.conv1 = PseudoConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        if use_inflated_groupnorm:
            self.norm2 = InflatedGroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
            
        self.dropout = torch.nn.Dropout(dropout)
        
        if use_pseudo_conv3d:
            self.conv2 = PseudoConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            if use_pseudo_conv3d:
                self.conv_shortcut = PseudoConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            else:
                self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if self.use_temporal_conv:
            self.temporal_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=False
            )
            
    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        bz, c, t, h, w = hidden_states.shape
        
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]
        
        # 添加首帧训练
        if temb.shape[0]==bz+1:
            temb_before = temb[:bz]
            temb_zero = temb[bz:].repeat(bz,1,1,1,1)

            hidden_states_zeros = torch.zeros_like(hidden_states).to(temb.device).to(temb.dtype)
            hidden_states_zeros[:,:,0,:,:] = hidden_states_zeros[:,:,0,:,:]+temb_zero.squeeze(-1)
            hidden_states_zeros[:,:,1:,:,:] = hidden_states_zeros[:,:,1:,:,:]+temb_before
            hidden_states = hidden_states + hidden_states_zeros
        else:
            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            import pdb; pdb.set_trace()
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        if self.use_temporal_conv:
            output_tensor = self.temporal_conv(output_tensor)
            
        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))