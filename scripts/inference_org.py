import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import DataLoader, Dataset

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# from pytorch_lightning import seed_everything
import time

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


def setup_dist(local_rank):
    if dist.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    
class PromptDataset(Dataset):
    def __init__(self, prompt_file):
        f = open(prompt_file, 'r')
        self.prompts, self.line_idx = [], []
        for idx, line in enumerate(f.readlines()):
            l = line.strip('\n')
            if len(l) != 0:
                self.prompts.append(l)
                self.line_idx.append(idx)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return self.prompts[index], self.line_idx[index]
    

# POS_PROMPT=", film grain, dramatic and cinematic lighting,8k, , ultra quality, film grain, 8K UHD, masterpiece,  high detailed, extremely detailed, photorealistic,"
POS_PROMPT = ", ultra quality, film grain, 8K UHD, masterpiece,  high detailed, extremely detailed, photorealistic, dramatic and cinematic lighting"
NEG_PROMPT="low resolution, low quality, lowres, worst quality,  noisy, duplicate, repeat, double, ugly, obese, deformed, render, rendered, bad anatomy,  text, asain, Chinese, watermark, bad anatomy, bad hands, text, missing finger,extra fingers"

def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # savedir = f"samples/{Path(args.config).stem}-{time_str}"
    savedir = f"{args.output_path}/{Path(args.config).stem}-{time_str}"
    inference_config = OmegaConf.load(args.inference_config)

    config  = OmegaConf.load(args.config)
    samples = []
    
    # set random seed
    if args.seed != -1:
        # if args.ddp:
        #     seed = args.local_rank + args.seed
        # else:
        seed = args.seed
        # seed_everything(seed)
        # seed = args.seed
        # seed_everything(seed)
        torch.manual_seed(seed)
    else:
        torch.seed()


    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            savedir = f"{savedir}_{model_config.path.split('/')[-1].split('.')[0]}_cfg_{model_config.guidance_scale}_step_{model_config.steps}_vs_{args.video_scale}_{motion_module.split('/')[-3]}_{motion_module.split('/')[-1]}_seed_{args.seed}"
            if args.local_rank == 0:
                print(f'savedir: {savedir}')
                os.makedirs(savedir)
            ### >>> create validation pipeline >>> ###
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            if args.use_ip:
                unet_base = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
                if is_xformers_available(): unet_base.enable_xformers_memory_efficient_attention()
                noise_scheduler_base = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
                pipeline_base = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_path,
                    unet=unet_base, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler_base, safety_checker=None,
                )
                pipeline_base.enable_vae_slicing()
                
                clip_image_processor = CLIPImageProcessor()
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_pretrained_model_path)

                pipeline = AnimationPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                    scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                    image_encoder=image_encoder
                ).to("cuda")
                
                pipeline.to("cuda")
                unet_base.to("cuda")
                
            else:
                pipeline = AnimationPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                    scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                ).to("cuda")

                
            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location="cpu")
            if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
            if 'state_dict' in motion_module_state_dict: 
                new_motion_module_state_dict = {}
                for k,v in motion_module_state_dict['state_dict'].items():
                    new_motion_module_state_dict[k.replace('module.', '')] = v
                missing, unexpected = pipeline.unet.load_state_dict(new_motion_module_state_dict, strict=True)
            else:
                missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            print (f'missing: {missing}')
            print (f'unexpected: {unexpected}')
            assert len(unexpected) == 0
            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)
                    
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    if args.use_ip:
                        pipeline_base.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    if args.use_ip:
                        pipeline_base.unet.load_state_dict(converted_unet_checkpoint, strict=True)
                    
                    # import pdb; pdb.set_trace()
                    # pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    # if args.use_ip:
                    #     pipeline_base.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)
                        if args.use_ip:
                            pipeline_base = convert_lora(pipeline_base, state_dict, alpha=model_config.lora_alpha)
                            
                            
            pipeline.to("cuda")
            if args.use_ip:
                pipeline_base.to("cuda")
            ### <<< create validation pipeline <<< ###
            prompts      = args.prompt
            prompt_file  = None
            if prompts.endswith(".txt"):
                prompt_file = prompts
            
            # initialize your dataset
            dataset = PromptDataset(prompt_file)
            if prompt_file is not None:
                cmd = f"cp {prompt_file} {savedir}"
                os.system(cmd)
                
            # initialize the DistributedSampler
            sampler = DistributedSampler(dataset)
            
            # initialize the dataloader
            dataloader = DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False
            )

            start = time.time()  
            config[config_key].random_seed = []
            
            generator = torch.Generator(device="cuda")
            generator.manual_seed(seed)
            
            for step, batch in enumerate(dataloader):
                prompt_list, row_index = batch
                prompt_list = list(prompt_list)
                refined_prompt_list = [p+POS_PROMPT for p in prompt_list]
                n_prompts    = list(NEG_PROMPT) * len(refined_prompt_list)
                
                for prompt_idx, (prompt, n_prompt) in enumerate(zip(refined_prompt_list, n_prompts)):
                    config[config_key].random_seed.append(torch.initial_seed())

                    print(f"current seed: {torch.initial_seed()}")
                    print(f"sampling {prompt} ...")
                    with torch.no_grad(), torch.autocast("cuda"):
                        
                        if args.use_ip:
                            first_images = pipeline_base(
                                prompt,
                                negative_prompt     = n_prompt,
                                height              = 512,
                                width               = 512,
                                num_inference_steps = 50,
                                guidance_scale      = 7.5,
                                generator           = generator
                            ).images
                            clip_image = clip_image_processor(images=first_images[0], return_tensors="pt").pixel_values
                            
                                
                            sample = pipeline(
                                prompt,
                                negative_prompt     = n_prompt,
                                num_inference_steps = model_config.steps,
                                guidance_scale      = model_config.guidance_scale,
                                width               = args.W,
                                height              = args.H,
                                video_length        = args.L,
                                video_scale         = args.video_scale,
                                use_ip_cross_attention = args.use_ip,
                                condition_images=clip_image,
                            ).videos
                            
                        else:
                            
                            sample = pipeline(
                                prompt,
                                negative_prompt     = n_prompt,
                                num_inference_steps = model_config.steps,
                                guidance_scale      = model_config.guidance_scale,
                                width               = args.W,
                                height              = args.H,
                                video_length        = args.L,
                                video_scale         = args.video_scale
                            ).videos
                        samples.append(sample)

                    prompt_str = prompt_list[prompt_idx].replace("/", " ")
                    video_idx = row_index[prompt_idx].item()
                    save_name = f'{video_idx}_{prompt_str}'
                    
                    # prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                    save_videos_grid(sample, f"{savedir}/sample/{save_name}.gif")
                    print(f"save to {savedir}/sample/{save_name}.gif")
                    if args.use_ip:
                        first_images[0].save(f'{savedir}/sample/{save_name}.png')
                        
                    sample_idx += 1

    # samples = torch.concat(samples)
    # save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")
    
    if args.local_rank==0:
        OmegaConf.save(config, f"{savedir}/config.yaml")
    
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output_path")
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
    parser.add_argument("--local_rank",            type=int, help="is used for pytorch ddp mode", default=0)
    parser.add_argument("--ddp",                   action='store_true', help="whether use pytorch ddp mode for parallel sampling (recommend for multi-gpu case)", default=False)
    parser.add_argument("--gpu_id",                type=int, help="choose a specific gpu", default=0)
    
    # sampling args
    parser.add_argument("--n_samples",  type=int, help="how many samples for each text prompt", default=1)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--seed",       type=int, default=-1, help="fix a seed for randomness (If you want to reproduce the sample results)")
    
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--video_scale", type=float, default=0.0)
    
    parser.add_argument("--use_ip",                   action='store_true', help="whether use ip_cross_attention", default=False)
    parser.add_argument("--image_pretrained_model_path",             type=str, default='image_pretrained_model_path')
    
    args = parser.parse_args()
    args.is_master = args.local_rank == 0
    args.device = torch.cuda.device(args.local_rank)
    
    # set device
    if args.ddp:
        setup_dist(args.local_rank)
        args.n_samples = math.ceil(args.n_samples / dist.get_world_size())
        gpu_id = None
    else:
        gpu_id = args.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    
    print (args)
    main(args)
