import os, io, csv, math, random
import numpy as np
from einops import rearrange, repeat
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print
from torch.utils.data import ConcatDataset, ChainDataset
from .majic_transformes import TXAugmentation
from transformers import CLIPImageProcessor
from PIL import Image
import random
import torchvision.transforms as T
import traceback
import glob
from tqdm import tqdm
import pyarrow as pa
import typing
import cv2

def get_moved_area_mask(frames, move_th=5, th=-1):
    ref_frame = frames[0] 
    # Convert the reference frame to gray
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = ref_gray
    # Initialize the total accumulated motion mask
    total_mask = np.zeros_like(ref_gray)

    # Iterate through the video frames
    for i in range(1, len(frames)):
        frame = frames[i]
        # Convert the frame to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the reference frame and the current frame
        diff = cv2.absdiff(ref_gray, gray)
        #diff += cv2.absdiff(prev_gray, gray)

        # Apply a threshold to obtain a binary image
        ret, mask = cv2.threshold(diff, move_th, 255, cv2.THRESH_BINARY)

        # Accumulate the mask
        total_mask = cv2.bitwise_or(total_mask, mask)

        # Update the reference frame
        prev_gray = gray

    contours, _ = cv2.findContours(total_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    ref_mask = np.zeros_like(ref_gray)
    ref_mask = cv2.drawContours(ref_mask, contours, -1, (255, 255, 255), -1)
    for cnt in contours:
        cur_rec = cv2.boundingRect(cnt)
        rects.append(cur_rec) 

    #rects = merge_overlapping_rectangles(rects)
    mask = np.zeros_like(ref_gray)
    if th < 0:
        h, w = mask.shape
        th = int(h*w*0.005)
    for rect in rects:
        x, y, w, h = rect
        if w*h < th:
            continue
        #ref_frame = cv2.rectangle(ref_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        mask[y:y+h, x:x+w] = 255
    return mask


def get_proper_resize_size(img, size):
    from PIL import Image
    if isinstance(size, int):
        size = (size, size)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    W, H = img.size
    h, w = size
    r = max(h / H, w / W)
    return int(H * r), int(W * r)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            clip_image_size=224,
            dynamic_fps=False,
            add_first_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'rU', newline="\n") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.dynamic_fps     = dynamic_fps
        self.add_first_image = add_first_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.video4flow_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
        ])
        
        self.clip_image_processor = transforms.Compose([
            transforms.Resize(clip_image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(clip_image_size),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        self.clip_image_processor_ori = CLIPImageProcessor()
        
    
    def get_batch(self, idx):
        # print (f'{idx}-get_batch-video')
        video_dict = self.dataset[idx]
        videoid, name = video_dict['videoid'], video_dict['name']
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        sample_stride = self.sample_stride

        if self.dynamic_fps:
            sample_stride = random.randint(2,24)
            # sample_stride = random.randint(2,10)
            # sample_stride = 4
        
        try:
            video_reader = VideoReader(video_dir)
        except Exception as e:
            print (f'Error: {e}, remove: {video_dir}')
            # if os.path.exists(video_dir): os.remove(video_dir)
            raise
        # print (f'{idx}-{video_dir}-{video_reader}')
        video_length = len(video_reader)
        # print (f'1-{idx}-{self.sample_stride}-{self.dynamic_fps}')
        # print (f'2-{idx}-{self.sample_stride}')
        
        if not self.is_image:
            framelst = list(range(0, len(video_reader), self.sample_stride))
            # print (f'{idx}-{framelst}') 
            if len(framelst)<self.sample_n_frames:
                sample_stride = len(video_reader)//(self.sample_n_frames+1)
                framelst = list(range(0, len(video_reader), sample_stride))
            if len(framelst)>self.sample_n_frames:
                start_idx = random.randint(0,len(framelst)-self.sample_n_frames)
            else:
                start_idx = 0
            batch_index = framelst[start_idx:start_idx+self.sample_n_frames]
                    
            # clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            # start_idx   = random.randint(0, video_length - clip_length)
            # batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        # print (f'3-{idx}-{self.sample_stride}-{batch_index}')
        
        video_images = video_reader.get_batch(batch_index)
        pixel_values = torch.from_numpy(video_images.asnumpy()).permute(0, 3, 1, 2).contiguous()
        

        # mask = get_moved_area_mask(pixel_values.permute([0,2,3,1]).numpy())
        # ratio = np.sum(mask==255)/(mask.shape[0]*mask.shape[1])
        
        # if ratio > 0.99:
        #     return self.get_batch(random.randint(0, video_length-1))
        
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name, video_images, sample_stride

    def get_first_image_condition(self, images):
        # pixel_values = torch.from_numpy(images.asnumpy()).permute(0, 3, 1, 2).contiguous()
        index = random.randint(0, images.asnumpy().shape[0]-1)
        pil_image = Image.fromarray(np.uint8(images.asnumpy()[index]))
        pixel_values = self.clip_image_processor_ori(images=pil_image, return_tensors="pt").pixel_values.squeeze(0)
        # pixel_values = self.clip_image_processor(pil_image)
        # print(f'111 {pixel_values}')
        return pixel_values
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            # while True:
                # try:
                    # pixel_values, name, video_images, sample_stride = self.get_batch(idx)
                    # break
                # except Exception as e:
                #     idx = random.randint(0, self.length-1)
            # print(f'1-{pixel_values.shape}')

            pixel_values, name, video_images, sample_stride = self.get_batch(idx)
            # import pdb;pdb.set_trace()
            pixel_values = self.pixel_transforms(pixel_values)
            # print(f'2-{pixel_values.shape}')
            if self.add_first_image:
                first_image = self.get_first_image_condition(video_images)
                video_frames = torch.from_numpy(video_images.asnumpy()).permute(0, 3, 1, 2)
                video_frames = self.video4flow_transforms(video_frames)
                sample = dict(pixel_values=pixel_values, text=name, ori_text=name, clip_images = first_image, fps=sample_stride, video_frames=video_frames)
            else:
                video_frames = torch.from_numpy(video_images.asnumpy()).permute(0, 3, 1, 2)
                video_frames = self.video4flow_transforms(video_frames)
                sample = dict(pixel_values=pixel_values, text=name, ori_text=name, fps=sample_stride, video_frames=video_frames)
            # print(f'{idx}-sample-{sample["pixel_values"].shape}')
            return sample
        except Exception as e:
            print(traceback.format_exc())
            new_idx = random.randint(0, self.length-1)
            return self.__getitem__(new_idx)


