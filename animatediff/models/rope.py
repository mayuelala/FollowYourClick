# RoPE
# https://zhuanlan.zhihu.com/p/642884818
# https://arxiv.org/pdf/2104.09864.pdf
import loguru
# NTK
# https://kexue.fm/archives/9706
# https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
# https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=b80b3f37

import torch
from typing import *
# from utils import bing_utils

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_freqs_cis(model, dim: int, seq_len: int, theta: float = 10000.0, attr_name='freq_cis'):
    if hasattr(model, attr_name):
        return getattr(model, attr_name)
    else:
        freq_cis = precompute_freqs_cis(dim, seq_len, theta)
        setattr(model, attr_name, freq_cis)
        return freq_cis


##
# try:
#     from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
# except:
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)


        # if bing_utils.global_config.dynamic_fps:
        #     assert bing_utils.global_config.inference_fps == int(4 * ratio), '如果不调大 fps condition，视频会等比例放慢'

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)


            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


from torch import nn
class RoPE(nn.Module):
    def __init__(self, dim, max_len=2048, base=10000, scale=0, alpha=0, dynamic_alpha=False, train_video_length=16, video_length=16): # alpha: NTK coeff
        super().__init__()
        self.max_len = max_len
        self.base = base
        self.dim = dim

        self.train_video_length = train_video_length
        self.video_length = video_length
        if scale:
            raise NotImplementedError

        if alpha:
            plan = 1
            if plan == 1:
                self.base = base * alpha ** (dim / (dim - 2))
                loguru.logger.debug(f'Update base to {self.base}')
            elif plan == 2:
                self.base = base * alpha
            else:
                raise Exception
            self.em = LlamaRotaryEmbedding(dim, max_len, self.base)
        else:
            self.em = LlamaRotaryEmbedding(dim, max_len, self.base)

        if dynamic_alpha:
            raise NotImplementedError
            ntk = 1
            ems = []
            for i in range(int(video_length / train_video_length)):
                curr_base = base * (ntk * video_length / train_video_length - ntk + 1) ** (dim / (dim - 2))
                em = LlamaRotaryEmbedding(dim, max_len, curr_base)
                ems.append(em)




    def forward(self, q, k):
        # q, k: b head n c
        assert len(q.shape) == 4
        assert len(k.shape) == 4
        assert q.shape[-2] == k.shape[-2], 'RoPE only supports self attention'

        seq_len = q.shape[-2]
        if seq_len > self.max_len:
            self.max_len = seq_len

        cos, sin = self.em(q, seq_len) # 传入的 q 只起device和dtype的作用
        q, k = apply_rotary_pos_emb(q, k, cos, sin, torch.arange(seq_len, dtype=torch.long, device=q.device).unsqueeze(0))
        if self.video_length > self.train_video_length:
            print('temporal log optimization')
            import math
            q = q * math.log(self.train_video_length, self.video_length)
        return q, k
        # from utils import bing_utils
        # bing_utils.temporary_modification_log('pos id 复制 看看会怎样')
        # return apply_rotary_pos_emb(q, k, cos, sin, torch.arange(16, dtype=torch.long, device=q.device).unsqueeze(0).repeat(1, seq_len//16))
        
if __name__ == '__main__':
    freqs_cis = precompute_freqs_cis(320, 31)
    apply_rotary_emb(torch.zeros(3, 10, 8, 31, 320), torch.zeros(3, 10, 8, 32, 320),  freqs_cis)

    em = LlamaRotaryEmbedding(320, 1)
    cos, sin = em(torch.zeros(10, 8, 31, 320), 31)
    apply_rotary_pos_emb(torch.zeros(10, 8, 31, 320), torch.zeros(10, 8, 31, 320), cos, sin, torch.arange(31).view(1, 31))