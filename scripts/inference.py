import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from einops import rearrange, repeat
import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from ip_adapter import MyIPAdapterPlus, MyIPAdapter

# from pytorch_lightning import seed_everything
import time
from pathlib import Path
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
        return {
            "prompt": self.prompts[index],
            "idx": self.line_idx[index],
        }

class PromptAndImgDataset(Dataset):
    def __init__(self, prompt_and_image_file):
        prompts_and_images = pd.read_excel(prompt_and_image_file)
        prompts = list(prompts_and_images["prompt"].values)
        image_paths = list(prompts_and_images["image"].values)
        assert len(prompts) == len(image_paths)
        self.prompts = []
        self.image_paths = []
        self.image_mask = []
        mask_root_path = '/teg_amai/share_1367250/mayuema/test_data/mask'
        # import pdb;pdb.set_trace()
        for prompt, image_path in zip(prompts, image_paths):
            if os.path.exists(image_path):
                self.prompts.append(prompt)
                self.image_paths.append(image_path)
                original_path = Path(image_path)
                image_mask_path = os.path.join(mask_root_path, original_path.parts[-2]+'_mask', original_path.stem + original_path.suffix)
                # image_mask_path = os.path.join(str(original_path.parent) + '_mask', original_path.stem + original_path.suffix)
                self.image_mask.append(image_mask_path)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "idx": idx,
            "image": self.image_paths[idx],
            "mask": self.image_mask[idx]
        }
    
# POS_PROMPT = " "
# POS_PROMPT=", film grain, dramatic and cinematic lighting,8k, , ultra quality, film grain, 8K UHD, masterpiece,  high detailed, extremely detailed, photorealistic,"
POS_PROMPT = ", ultra quality, film grain, 8K UHD, masterpiece,  high detailed, extremely detailed, photorealistic, dramatic and cinematic lighting"

NEG_PROMPT="low resolution, low quality, lowres, worst quality,  noisy, duplicate, repeat, double, ugly, obese, deformed, render, rendered, bad anatomy,  text, watermark, bad anatomy, bad hands, text, missing finger,extra fingers"
# NEG_PROMPT = " "


def main(args):
    # import pdb; pdb.set_trace()
    
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
            # savedir = f"{savedir}_{motion_module.split('/')[-1].split('.')[0]}_cfg_{model_config.guidance_scale}_step_{model_config.steps}_vs_{args.video_scale}_{motion_module.split('/')[-3]}_{motion_module.split('/')[-1]}_seed_{args.seed}"
            savedir = f"{savedir}_{motion_module.split('/')[-1].split('.')[0]}_{motion_module.split('/')[-1]}_seed_{args.seed}_fps_{args.fps}_flowctrl_{args.flw_ctrl}"
            if args.local_rank == 0:
                print(f'savedir: {savedir}')
                os.makedirs(savedir)
            ### >>> create validation pipeline >>> ###
            
            # model loading
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
            unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
            
            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False
            
            ip_adapter_model = None
            if args.use_ip != "":
                if args.use_ip == "vanilla":
                    ip_adapter_model = MyIPAdapter(unet, args.image_pretrained_model_path, "", args.local_rank, num_tokens=4)
                else:
                    ip_adapter_model = MyIPAdapterPlus(unet, args.image_pretrained_model_path, "", args.local_rank, num_tokens=16)
                unet.image_proj_model = ip_adapter_model.init_proj()
                image_encoder = ip_adapter_model.image_encoder
                
            # unet ckpt
            motion_module_state_dict = torch.load(# The above code is importing a module called
            # "motion_module" in Python.
            motion_module, map_location="cpu")
            if 'state_dict' in motion_module_state_dict:
                motion_module_state_dict = motion_module_state_dict['state_dict']
            new_motion_module_state_dict = {}
            for k,v in motion_module_state_dict.items():
                new_motion_module_state_dict[k.replace('module.', '')] = v
            missing, unexpected = unet.load_state_dict(new_motion_module_state_dict, strict=False)
            print (f'missing: {len(missing)}')
            print (f'unexpected: {len(unexpected)}')
            assert len(unexpected) == 0 
                
            # import pdb;pdb.set_trace()
            # if not args.use_fps_condition:
            #     unexpect_filt = []
            #     if len(unexpected) != 0:
            #         for key in unexpected:
            #             if 'fps' not in key:
            #                 unexpect_filt.append(key)                
            #     assert len(unexpect_filt) == 0
            # else:
            #     assert len(unexpected) == 0
            
            if not args.manually_input_image:
                unet_base = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
                if is_xformers_available(): unet_base.enable_xformers_memory_efficient_attention()
                unet_base.to("cuda")
                noise_scheduler_base = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
                pipeline_base = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_path,
                    unet=unet_base, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler_base, safety_checker=None,
                )
                pipeline_base.enable_vae_slicing()
                pipeline_base.to("cuda")
                
            pipeline = AnimationPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                ip_adapter=ip_adapter_model,
            ).to("cuda")
            
            # reload unet spaçtial part for image
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
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
                        base_state_dict, pipeline.unet.config,
                        need_img_embed_concat=inference_config.unet_additional_kwargs.use_first_frame_mask_condition_concat)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)
                            
            ### <<< create validation pipeline <<< ###
            
            if not args.manually_input_image:
                # initialize your dataset
                dataset = PromptDataset(args.file)
            else:
                dataset = PromptAndImgDataset(args.file)
            if args.file is not None:
                cmd = f"cp {args.file} {savedir}"
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
                prompt_list  = batch["prompt"]
                row_index = batch["idx"]
                if args.manually_input_image:
                    first_images_paths = batch["image"]
                    first_images_mask_paths = batch["mask"]
                    
                prompt_list = list(prompt_list)
                refined_prompt_list = [p+POS_PROMPT for p in prompt_list]
                n_prompts    = list(NEG_PROMPT) * len(refined_prompt_list)
                
                for prompt_idx, (prompt, n_prompt) in enumerate(zip(refined_prompt_list, n_prompts)):
                    config[config_key].random_seed.append(torch.initial_seed())

                    print(f"current seed: {torch.initial_seed()}")
                    print(f"sampling {prompt} ...")
                    with torch.no_grad(), torch.autocast("cuda"):
                        
                        # import pdb; pdb.set_trace()
                        
                        first_image_latents = None
                        if not args.manually_input_image:
                            first_images = pipeline_base(
                                prompt,
                                height              = 512,
                                width               = 512,
                                num_inference_steps = 50,
                                guidance_scale      = 8.,
                            ).images
                        else:
                            first_images = [Image.open(first_images_paths[prompt_idx]).convert('RGB')]
                            first_mask_images = [Image.open(first_images_mask_paths[prompt_idx]).convert('RGB')]
                            
                        first_images_list = []
                        first_images_mask_list = []
                        
                        pixel_transforms = transforms.Compose([
                            transforms.Resize(max(args.W, args.H)),
                            transforms.CenterCrop((args.H, args.W)),
                        ])
                        
                        cond_imgs = None
                        if args.use_ip != "":
                            cond_imgs = ip_adapter_model.clip_image_processor(images=first_images, return_tensors="pt").pixel_values

                        for img, mask in zip(first_images, first_mask_images):
                            if args.crop_method == 'resize_and_crop':
                                resize_target = max([args.W, args.H])
                                img = img.resize((resize_target, resize_target))
                                mask = mask.resize((resize_target, resize_target))

                                left, right = (resize_target - args.W) / 2, (resize_target + args.W) / 2
                                top, bottom = (resize_target - args.H) / 2, (resize_target + args.H) / 2
                                left, top = round(max(0, left)), round(max(0, top))
                                right, bottom = round(min(resize_target, right)), round(min(resize_target, bottom))
                                # import pdb;pdb.set_trace()
                                img = img.crop((left, top, right, bottom))
                                mask = mask.crop((left, top, right, bottom))

                            elif args.crop_method == 'crop':
                                img = pixel_transforms(img)
                            
                            first_frame_output = torch.from_numpy(np.array(img)).div(255) * 2 - 1
                            first_frame_output = rearrange(first_frame_output, "h w c -> c h w").unsqueeze(0)
                            first_frame_output = first_frame_output.to(dtype= vae.dtype, device=vae.device)
                            first_images_list.append(first_frame_output)

                            first_frame_mask_output = torch.from_numpy(np.array(mask))
                            first_frame_mask_output = rearrange(first_frame_mask_output, "h w c -> c h w").unsqueeze(0)
                            first_frame_mask_output = first_frame_mask_output.to(dtype= vae.dtype, device=vae.device)
                            first_images_mask_list.append(first_frame_mask_output)



                        first_images = torch.cat(first_images_list, dim=0).to(args.local_rank)
                        first_images_mask1 = torch.cat(first_images_mask_list, dim=0).to(args.local_rank)
                        
                    
                        first_image_latents = vae.encode(first_images).latent_dist
                        first_image_latents = first_image_latents.sample()
                        first_image_latents = first_image_latents * 0.18215


                        vae_scale_factor=8
                        first_images_mask_tmp = torch.nn.functional.interpolate(first_images_mask1, 
                                                                size=(first_images.size(-2) // vae_scale_factor,
                                                                        first_images.size(-1) // vae_scale_factor))[:,None]
                        first_images_mask = torch.clamp(first_images_mask_tmp, 0, 1)

                        # import pdb; pdb.set_trace()
                        
                        use_first_frame_mask_condition_concat_image_partial_mask = None
                        if args.mask_first_frame:
                            use_first_frame_mask_condition_concat_image_partial_mask = torch.rand_like(first_image_latents)[:1,:1,:,:].unsqueeze(0)
                            use_first_frame_mask_condition_concat_image_partial_mask = (use_first_frame_mask_condition_concat_image_partial_mask>0.5).to(first_image_latents.dtype)
                        
                        sample = pipeline(
                            prompt,
                            negative_prompt     = n_prompt,
                            num_inference_steps = model_config.steps,
                            guidance_scale      = model_config.guidance_scale,
                            width               = args.W,
                            height              = args.H,
                            video_length        = args.L,
                            video_scale         = args.video_scale,
                            use_first_frame_mask_condition_concat=inference_config.unet_additional_kwargs.use_first_frame_mask_condition_concat,
                            first_image_latents = first_image_latents,
                            use_first_image_as_init_latents = args.use_first_image_as_init_latents,
                            use_first_frame_mask_condition_concat_image_partial_mask = use_first_frame_mask_condition_concat_image_partial_mask,
                            use_fps_condition=args.use_fps_condition,
                            fps_tensor=torch.tensor([args.fps]),
                            use_ip_cross_attention = ip_adapter_model is not None,
                            condition_images=cond_imgs,
                            use_interpolate_noise = args.use_interpolate_noise,
                            use_first_frame_mask_condition_concat_zero_padding = args.use_first_frame_mask_condition_concat_zero_padding,
                            first_images_mask=first_images_mask,
                            flow_control=torch.tensor([args.flw_ctrl]),
                        ).videos
                        
                        first_images = (first_images - first_images.min()) / (first_images.max() - first_images.min())
                        first_images = first_images.detach().cpu()
                        # import pdb;pdb.set_trace()
                        if args.show_image:
                            show_mask = first_images_mask1[:,:,None].repeat(1, 1, args.L, 1, 1).cpu()
                            first_images_repeated = torch.unsqueeze(first_images, 2).repeat(1,1,args.L,1,1)
                            sample = torch.cat((show_mask //255., sample, first_images_repeated), dim=-1)
                            if args.mask_first_frame:
                                first_images_repeated_partial = pipeline.decode_latents(torch.unsqueeze(first_image_latents, 2).repeat(1,1,args.L,1,1)*use_first_frame_mask_condition_concat_image_partial_mask)
                                sample = torch.cat((sample, torch.from_numpy(first_images_repeated_partial)), dim=-1)
                                
                        samples.append(sample)

                    prompt_str = prompt_list[prompt_idx].replace("/", " ")
                    video_idx = row_index[prompt_idx].item()
                    save_name = f'{video_idx}_{prompt_str}'
                    
                    # prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                    
                    save_videos_grid(sample, f"{savedir}/sample/{save_name}.gif")
                    print(f"save to {savedir}/sample/{save_name}.gif")
                        
                    sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")
    
    if args.local_rank==0:
        OmegaConf.save(config, f"{savedir}/config.yaml")
    
    if args.ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output_path")
    parser.add_argument("--pretrained_model_path", type=str, default="pretrained_model_path/huggingface_models/stable-diffusion-v1-5") #"YOUR PATH/share_301124792/antonwang/cache/stable-diffusion-v1-5",)
    parser.add_argument("--image_pretrained_model_path", type=str, default="pretrained_model_path/gitpackages/huggingface_models/IP-Adapter/models/image_encoder",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
    parser.add_argument("--local_rank",            type=int, help="is used for pytorch ddp mode", default=0)
    parser.add_argument("--ddp",                   action='store_true', help="whether use pytorch ddp mode for parallel sampling (recommend for multi-gpu case)", default=False)
    parser.add_argument("--gpu_id",                type=int, help="choose a specific gpu", default=0)
    
    # sampling args
    parser.add_argument("--n_samples",  type=int, help="how many samples for each text prompt", default=1)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--seed",       type=int, default=-1, help="fix a seed for randomness (If you want to reproduce the sample results)")
    
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--video_scale", type=float, default=0.0)
    parser.add_argument("--use_fps_condition", action='store_true', help="", default=False)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--flw_ctrl", type=int, default=4)
    parser.add_argument("--use_ip", type=str, default="")
    parser.add_argument("--manually_input_image", action='store_true', help="", default=False)
    parser.add_argument("--show_image", action='store_true', help="", default=True)
    parser.add_argument("--crop_method", type=str, choices=['resize_and_crop', 'crop'], help="", default='crop')
    parser.add_argument("--use_first_image_as_init_latents", action='store_true', help="", default=False)
    parser.add_argument("--mask_first_frame", action='store_true', help="", default=False)
    parser.add_argument("--use_interpolate_noise", action='store_true', help="", default=False)
    parser.add_argument("--use_first_frame_mask_condition_concat_zero_padding", action='store_true', help="", default=True)
    
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
