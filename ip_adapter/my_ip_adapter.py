import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
# from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import is_torch2_available

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MyIPAdapter:
    def __init__(self, unet, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.unet = unet

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device)
        self.image_encoder.requires_grad_(False)
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device)
        return image_proj_model
    
    def get_ip_adapter_state_dict(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        return state_dict
    
    def load_ip_adapter(self, unet = None, use_unet_image_proj_model = False):
        state_dict = self.get_ip_adapter_state_dict()
        
        if unet is not None:
            unet_state_dict = unet.state_dict()
        else:
            unet_state_dict = self.unet.state_dict()
        
        if use_unet_image_proj_model:
            unet_state_dict['image_proj_model.proj.weight'] = state_dict['image_proj']['proj.weight']
            unet_state_dict['image_proj_model.proj.bias'] = state_dict['image_proj']['proj.bias']
            unet_state_dict['image_proj_model.norm.weight'] = state_dict['image_proj']['norm.weight']
            unet_state_dict['image_proj_model.norm.bias'] = state_dict['image_proj']['norm.bias']
        else:
            missing_image_proj_model, unexpected_image_proj_model = self.image_proj_model.load_state_dict(state_dict["image_proj"])
            print("load image_proj_model: missing keys: {}, unexpected keys: {}".format(len(missing_image_proj_model), len(unexpected_image_proj_model)))
        
        
        ip_keys = []
        for k, v in state_dict['ip_adapter'].items():
            print(k, v.shape)
            ip_keys.append(k)
        
        model_replace_keys = []
        for k in unet_state_dict:
            if '_ip' in k:
                print(k, unet_state_dict[k].shape)
                model_replace_keys.append(k)
                
        for k1, k2 in zip(model_replace_keys, ip_keys):
            print (f'replace {k1} with params of {k2}')
            assert unet_state_dict[k1].shape == state_dict['ip_adapter'][k2].shape
            unet_state_dict[k1] = state_dict['ip_adapter'][k2]
        
        if unet is not None:
            missing_unet, unexpected_unet = unet.load_state_dict(unet_state_dict, strict=False)
        else:
            missing_unet, unexpected_unet = self.unet.load_state_dict(unet_state_dict, strict=False)
        print("load ip_adapter to unet: missing keys: {}, unexpected keys: {}".format(len(missing_unet), len(unexpected_unet)))
        
        return missing_unet, unexpected_unet
    
    @torch.no_grad()
    def get_image_clip_feat(self, input_image=None):
        if isinstance(input_image, Image.Image):
            input_image = [input_image]
            input_image = self.clip_image_processor(images=input_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(input_image.to(self.device)).image_embeds
        uncond_clip_image_embeds = torch.zeros_like(clip_image_embeds)
        return clip_image_embeds, uncond_clip_image_embeds
    
    @torch.inference_mode()
    def get_image_embeds(self, input_image=None, clip_image_embeds=None, image_proj_model=None):
        if input_image is not None:
            if isinstance(input_image, Image.Image):
                input_image = [input_image]
                input_image = self.clip_image_processor(images=input_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(input_image.to(self.device)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device)
        if image_proj_model is None:
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        else:
            image_prompt_embeds = image_proj_model(clip_image_embeds)
        if image_proj_model is None:
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        else:
            uncond_image_prompt_embeds = image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    # def generate(
    #     self,
    #     pil_image=None,
    #     clip_image_embeds=None,
    #     prompt=None,
    #     negative_prompt=None,
    #     scale=1.0,
    #     num_samples=4,
    #     seed=None,
    #     guidance_scale=7.5,
    #     num_inference_steps=30,
    #     **kwargs,
    # ):
    #     self.set_scale(scale)

    #     if pil_image is not None:
    #         num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
    #     else:
    #         num_prompts = clip_image_embeds.size(0)

    #     if prompt is None:
    #         prompt = "best quality, high quality"
    #     if negative_prompt is None:
    #         negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    #     if not isinstance(prompt, List):
    #         prompt = [prompt] * num_prompts
    #     if not isinstance(negative_prompt, List):
    #         negative_prompt = [negative_prompt] * num_prompts

    #     image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
    #         pil_image=pil_image, clip_image_embeds=clip_image_embeds
    #     )
    #     bs_embed, seq_len, _ = image_prompt_embeds.shape
    #     image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
    #     image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
    #     uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
    #     uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

    #     with torch.inference_mode():
    #         prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
    #             prompt,
    #             device=self.device,
    #             num_images_per_prompt=num_samples,
    #             do_classifier_free_guidance=True,
    #             negative_prompt=negative_prompt,
    #         )
    #         prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
    #         negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

    #     generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
    #     images = self.pipe(
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         guidance_scale=guidance_scale,
    #         num_inference_steps=num_inference_steps,
    #         generator=generator,
    #         **kwargs,
    #     ).images

    #     return images


class MyIPAdapterPlus(MyIPAdapter):
    """IP-Adapter with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device)
        return image_proj_model
    
    def load_ip_adapter(self, unet = None, use_unet_image_proj_model = False):
        state_dict = self.get_ip_adapter_state_dict()
                
        if use_unet_image_proj_model:
            unet.image_proj_model = self.init_proj()
            missing_image_proj_model, unexpected_image_proj_model = unet.image_proj_model.load_state_dict(state_dict["image_proj"])
            print("load image_proj_model: missing keys: {}, unexpected keys: {}".format(len(missing_image_proj_model), len(unexpected_image_proj_model)))
        else:
            missing_image_proj_model, unexpected_image_proj_model = self.image_proj_model.load_state_dict(state_dict["image_proj"])
            print("load image_proj_model: missing keys: {}, unexpected keys: {}".format(len(missing_image_proj_model), len(unexpected_image_proj_model)))
        
        if unet is not None:
            unet_state_dict = unet.state_dict()
        else:
            unet_state_dict = self.unet.state_dict()
            
        ip_keys = []
        for k, v in state_dict['ip_adapter'].items():
            print(k, v.shape)
            ip_keys.append(k)
        
        model_replace_keys = []
        for k in unet_state_dict:
            if '_ip' in k:
                print(k, unet_state_dict[k].shape)
                model_replace_keys.append(k)
                
        for k1, k2 in zip(model_replace_keys, ip_keys):
            print (f'replace {k1} with params of {k2}')
            assert unet_state_dict[k1].shape == state_dict['ip_adapter'][k2].shape
            unet_state_dict[k1] = state_dict['ip_adapter'][k2]
        
        if unet is not None:
            missing_unet, unexpected_unet = unet.load_state_dict(unet_state_dict, strict=False)
        else:
            missing_unet, unexpected_unet = self.unet.load_state_dict(unet_state_dict, strict=False)
        print("load ip_adapter to unet: missing keys: {}, unexpected keys: {}".format(len(missing_unet), len(unexpected_unet)))
        
        return missing_unet, unexpected_unet
    
    @torch.no_grad()
    def get_image_clip_feat(self, input_image=None):
        if isinstance(input_image, Image.Image):
            input_image = [input_image]
            input_image = self.clip_image_processor(images=input_image, return_tensors="pt").pixel_values
        input_image = input_image.to(self.device)
        clip_image_embeds = self.image_encoder(input_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(input_image), output_hidden_states=True
        ).hidden_states[-2]
        return clip_image_embeds, uncond_clip_image_embeds
    
    @torch.inference_mode()
    def get_image_embeds(self, input_image=None, clip_image_embeds=None, image_proj_model=None):
        if isinstance(input_image, Image.Image):
            input_image = [input_image]
            input_image = self.clip_image_processor(images=input_image, return_tensors="pt").pixel_values
        input_image = input_image.to(self.device)
        clip_image_embeds = self.image_encoder(input_image, output_hidden_states=True).hidden_states[-2]
        if image_proj_model is None:
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        else:
            image_prompt_embeds = image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(input_image), output_hidden_states=True
        ).hidden_states[-2]
        if image_proj_model is None:
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        else:
            uncond_image_prompt_embeds = image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

