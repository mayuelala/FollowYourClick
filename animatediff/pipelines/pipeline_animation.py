# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
from packaging import version
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer

from diffusers.utils import is_accelerate_available
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from ..models.unet import UNet3DConditionModel
from ..utils.util import preprocess_image, slerp

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_encoder: CLIPVisionModelWithProjection=None,
        text_encoder_2: T5EncoderModel=None,
        tokenizer_2: T5Tokenizer=None,
        ip_adapter = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            ip_adapter = ip_adapter,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.image_encoder]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def _encode_prompt_2(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds: Optional[torch.FloatTensor] = None, negative_prompt_embeds: Optional[torch.FloatTensor] = None,):
    #     self,
    #     prompt,
    #     do_classifier_free_guidance=True,
    #     num_images_per_prompt=1,
    #     device=None,
    #     negative_prompt=None,
    #     prompt_embeds: Optional[torch.FloatTensor] = None,
    #     negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    #     clean_caption: bool = False,
    # ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        if prompt_embeds is None:
            text_inputs = self.tokenizer_2(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            attention_mask = text_inputs.attention_mask.to(device)

            prompt_embeds = self.text_encoder_2(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder_2 is not None:
            dtype = self.text_encoder_2.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer_2(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder_2(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None

        # return prompt_embeds, negative_prompt_embeds
            
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
        return text_embeddings
    
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    
    def prepare_latents(
        self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None,
        init_latents = None, init_image=None, use_residual_noise=False, base_lambda=0.9, k = 64, use_interpolate_noise = True, use_add_noise=False, 
        first_images_mask = None):
        
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if init_image is not None and init_latents is None:
            if isinstance(init_image, Image.Image):
                image = preprocess_image(init_image, height, width)
            print(image.shape)
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )
            image = image.to(device=device, dtype=dtype)
            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)   
                init_latents = init_latents * 0.18215
            
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                
                if use_interpolate_noise:
                    latents = (latents[:,:,:1]).repeat(1,1,latents.shape[2],1,1)
                    # latents = torch.nn.functional.interpolate(latents[:,:,:2], size=latents.shape[2:], mode='trilinear')
                    # new_latents = [
                    #     slerp(i, latents[:,:,:1], latents[:,:,:-1]) for i in torch.arange(latents.shape[2]) / latents.shape[2]
                    #     ]
                    # latents = torch.cat(new_latents, dim=2).to(device)
                    
                if init_latents is not None:
                    for i in range(video_length):
                        # I just feel dividing by 30 yield stable result but I don't know why
                        # gradully reduce init alpha along video frames (loosen restriction)
                        init_alpha = (video_length - float(i)) / video_length / k
                        print(f'{i}-{init_alpha}')
                        mask = first_images_mask[:, :,i].repeat(1, 4, 1, 1)
                        latents[:, :, i, :, :] = init_latents * init_alpha + latents[:, :, i, :, :] * (1 - init_alpha) 
                if use_residual_noise:
                    latents_base = latents[:,:,0].unsqueeze(2).repeat(1,1,video_length,1,1)
                    # latents = latents[:,:,0].unsqueeze(2).repeat(1,1,num_frames,1,1)
                    latents = base_lambda**0.5*latents_base + (1-base_lambda)**0.5*latents
                    latents[:,:,0] = latents_base[:,:,0]
            
            
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
            
            if use_interpolate_noise:
                # latents = (latents[:,:,:1]).repeat(1,1,latents.shape[2],1,1)
                # latents = torch.nn.functional.interpolate(latents[:,:,:2], size=latents.shape[2:], mode='trilinear')
                pass
                    
            if init_latents is not None:
                for i in range(video_length):
                    # I just feel dividing by 30 yield stable result but I don't know why
                    # gradully reduce init alpha along video frames (loosen restriction)
                    init_alpha = (video_length - float(i)) / video_length / k
                    print(f'{i}-{init_alpha}')
                    latents[:, :, i, :, :] = init_latents * init_alpha + latents[:, :, i, :, :] * (1 - init_alpha)


        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_image_prompt(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image_embeds = self.image_encoder(pil_image.to(self.device)).image_embeds
        uncond_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
        return clip_image_embeds, uncond_image_prompt_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        use_first_frame_condition: bool=False,
        use_first_frame_condition_concat: bool=False,
        use_first_frame_mask_condition_concat: bool = False, # from paper EMU VIDEO: Factorizing Text-to-Video Generation by Explicit Image Conditioning
        use_first_frame_mask_condition_concat_image_partial_mask = None,
        first_image_latents=None,
        use_first_image_as_init_latents=False,
        video_scale=0,
        use_ip_cross_attention = False,
        condition_images=None,
        use_uncond_images=False,
        use_camera_motion_condition=False,
        camera_movement_type=None,
        use_text_encoder_2=False,
        use_uncond_text_2=False,
        use_fps_condition=False,
        fps_tensor=None,
        use_interpolate_noise=False,
        first_images_mask=None,
        flow_control=None,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        text_embeddings_2=None
        if use_text_encoder_2:
            text_embeddings_2 = self._encode_prompt_2(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            # print(f'{prompt}-{text_embeddings_2}')
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps


        first_images_mask_final = None
        if first_images_mask is not None:
            # mask 1st frame
            # masks_other_frame = torch.zeros_like(first_images_mask[:,:, 0:1]).repeat(1, 1,  video_length - 1, 1, 1).to(first_images_mask.device)
            # first_images_mask_final_tmp = torch.cat([first_images_mask[:,:, 0:1], masks_other_frame], dim=2)
            
            #==========
            # mask all
            first_images_mask_final_tmp = first_images_mask[:,:, 0:1].repeat(1, 1, video_length, 1, 1)
            
            #==========
            first_images_mask_final = torch.clamp(first_images_mask_final_tmp, 0, 1)

        # Prepare latent variables

        num_channels_latents = self.unet.in_channels
        if use_first_image_as_init_latents:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
                init_latents = first_image_latents,
                use_interpolate_noise = use_interpolate_noise,
                init_image=None,
                first_images_mask=first_images_mask_final
            )
        else:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
                use_interpolate_noise = use_interpolate_noise,
            )
            
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        image_prompt_embeds, uncond_image_prompt_embeds = None, None
        if use_ip_cross_attention:
            # image_prompt_embeds, uncond_image_prompt_embeds = self._encode_image_prompt(condition_images)
            image_prompt_embeds, uncond_image_prompt_embeds = self.ip_adapter.get_image_clip_feat(input_image=condition_images)
            if use_uncond_images:
                image_prompt_embeds = uncond_image_prompt_embeds.clone()
        
        # Train / Val is None, Test is true



        with torch.no_grad(), torch.autocast("cuda"):
            # Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if use_first_frame_condition:
                        latents[:,:,0,:,:] = first_image_latents
                    elif use_first_frame_mask_condition_concat:
                        first_frame_of_latents_block = torch.zeros_like(latents)
                        first_frame_of_latents_block[:,:,0,:,:] = first_image_latents
                        mask_block = torch.zeros_like(latents)[:,:1]
                        mask_block[:,:,0,:,:] = 1
                        if use_first_frame_mask_condition_concat_image_partial_mask is not None:
                            first_frame_of_latents_block[:,:,0,:,:] = first_frame_of_latents_block[:,:,0,:,:]*use_first_frame_mask_condition_concat_image_partial_mask
                        
                        if first_images_mask is not None:
                            latents_use_first_frame_mask_condition_concat = torch.cat((latents, first_images_mask_final, first_frame_of_latents_block), dim=1)
                        else:
                            latents_use_first_frame_mask_condition_concat = torch.cat((latents, mask_block, first_frame_of_latents_block), dim=1)
                            
                    
                    # expand the latents if we are doing classifier free guidance
                    if use_first_frame_mask_condition_concat:
                        latent_model_input = torch.cat([latents_use_first_frame_mask_condition_concat] * 2) if do_classifier_free_guidance else latents_use_first_frame_mask_condition_concat
                    else:
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, 
                                        encoder_hidden_states=text_embeddings, 
                                        use_first_frame_condition=use_first_frame_condition,
                                        use_first_frame_condition_concat=use_first_frame_condition_concat,
                                        reference_images_latent=torch.cat([first_image_latents] * 2) if (do_classifier_free_guidance and first_image_latents is not None) else first_image_latents,
                                        use_ip_cross_attention = use_ip_cross_attention,
                                        reference_images_clip_feat=torch.cat([uncond_image_prompt_embeds, image_prompt_embeds]) if # The above code is not valid Python code. It seems to be a placeholder or a comment.
                                        do_classifier_free_guidance and use_ip_cross_attention else image_prompt_embeds,
                                        use_camera_motion_condition=use_camera_motion_condition,
                                        camera_movement_type_tensor=torch.cat([camera_movement_type] * 2)  if (do_classifier_free_guidance and camera_movement_type is not None) else camera_movement_type,
                                        use_text_encoder_2=use_text_encoder_2,
                                        encoder_hidden_states_2=text_embeddings_2,
                                        use_fps_condition=use_fps_condition,
                                        fps_tensor=torch.cat([fps_tensor] * 2)  if (do_classifier_free_guidance and fps_tensor is not None) else fps_tensor,
                                        flow_control=torch.cat([flow_control] * 2)  if (do_classifier_free_guidance and flow_control is not None) else flow_control,
                                        ).sample.to(dtype=latents_dtype)
                    # noise_pred = []
                    # import pdb
                    # pdb.set_trace()
                    # for batch_idx in range(latent_model_input.shape[0]):
                    #     noise_pred_single = self.unet(latent_model_input[batch_idx:batch_idx+1], t, encoder_hidden_states=text_embeddings[batch_idx:batch_idx+1]).sample.to(dtype=latents_dtype)
                    #     noise_pred.append(noise_pred_single)
                    # noise_pred = torch.cat(noise_pred)
                    if video_scale>0:
                        bsz = latents.shape[0]
                        f = latents.shape[2]
                        # 逐帧预测
                        latent_model_input_single_frame = rearrange(latent_model_input, 'b c f h w -> (b f) c h w').unsqueeze(2)
                        text_embeddings_single_frame = torch.cat([text_embeddings] * f, dim=0)
                        latent_model_input_single_frame = latent_model_input_single_frame.chunk(2, dim=0)[0]
                        text_embeddings_single_frame = text_embeddings_single_frame.chunk(2, dim=0)[0]
                        noise_pred_single_frame_uncond = self.unet(
                                    latent_model_input_single_frame,
                                    t,
                                    encoder_hidden_states = text_embeddings_single_frame,
                                    ).sample
                        noise_pred_single_frame_uncond = rearrange(noise_pred_single_frame_uncond.squeeze(2), '(b f) c h w -> b c f h w', f=f)
                        
                    # perform guidance
                    if do_classifier_free_guidance:
                        if video_scale > 0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_single_frame_uncond + video_scale * (
                                noise_pred_uncond - noise_pred_single_frame_uncond
                            ) + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                        else:    
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # Post-processing
        #===========================================================
    
        #===========================================================                     
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
