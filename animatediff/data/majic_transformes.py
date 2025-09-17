import math
import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def myrotate(image, angle):
    image_height, image_width = image.shape[0:2]
    image_orig = np.copy(image)
    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    )
    return image_rotated_cropped

###################################################

import random
import torch
import einops

LEFT = 'pan left'
RIGHT = 'pan right'
UP = 'pan up'
DOWN = 'pan down'
ZOOM_IN = 'zoom in'
ZOOM_OUT = 'zoom out'
ROTATE_LEFT = 'rotate anticlockwise'
ROTATE_RIGHT = 'rotate clockwise'

MOTION_TYPES = [LEFT, RIGHT, UP, DOWN, ZOOM_IN, ZOOM_OUT, ROTATE_LEFT, ROTATE_RIGHT]


from torchvision import transforms
from torchvision import transforms as T
from einops import rearrange
from torchvision import transforms
from torchvision.transforms import _functional_video as VF
import torchvision.transforms._transforms_video as transforms_video
import torchvision.transforms._transforms_video as transforms_video

def get_proper_resize_size_by_size(curr_size, size):
    H, W = curr_size
    h, w = size
    r = max(h / H, w / W)
    return int(H * r), int(W * r)


class ResizeCenterCropVideo:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, vid):
        # resolution = get_proper_resize_size_by_size(vid.shape[-2:], self.resolution)
        transform = transforms.Compose([
            transforms.Resize(self.resolution[1]),
            transforms.CenterCrop(self.resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        vid = transform(vid)
            # lambda x: VF.resize(x, (resolution, resolution) if isinstance(resolution, int) else tuple(resolution), interpolation_mode="bilinear"),

            
        
        return vid

class TXAugmentation:

    def __init__(self, resolution, up_down_ratio=0.2, left_right_ratio=0.2, zoom_ratio=0.2, max_degree=30, num_frames=16):
        self.resolution = resolution
        self.num_frames = num_frames
        self.resize_centercrop = ResizeCenterCropVideo(resolution)
        self.up_down_ratio = up_down_ratio
        self.left_right_ratio = left_right_ratio
        self.zoom_ratio = zoom_ratio
        self.max_degree = max_degree


    def up_down(self, vid, d):
        # vid = bing_utils.resize_crop(vid, self.resolution) # 左右平移可以先resize
        h, _ = vid[0].shape[-2:]
        move_ratio = self.up_down_ratio
        cropped = int(move_ratio * h)
        step = cropped // self.num_frames

        if d == 1: # down
            x1s = torch.arange(0, cropped, step)
        else: # up
            x1s = torch.arange(cropped, 0, -step)
        ret = []
        for idx in range(self.num_frames):
            ret.append(vid[:, idx:idx+1, x1s[idx]:x1s[idx] + h - cropped])
        ret = torch.cat(ret, dim=1)
        ret = einops.rearrange(ret, 'c t h w -> t c h w', t=self.num_frames)
        return self.resize_centercrop(ret)

    def left_right(self, vid, d):
        # vid = bing_utils.resize_crop(vid, self.resolution) # 左右平移可以先resize
        # vid = self.resize_centercrop(vid)
        h, w = vid.shape[-2:]
        move_ratio = self.left_right_ratio
        cropped = int(move_ratio * w)
        step = cropped // self.num_frames

        if d == 1:
            x1s = torch.arange(0, cropped, step)
        else: # left
            x1s = torch.arange(cropped, 0, -step)
        ret = []
        for idx in range(self.num_frames):
            ret.append(vid[:, idx:idx+1, :, x1s[idx]:x1s[idx] + w - cropped])
        ret = torch.cat(ret, dim=1)
        ret = einops.rearrange(ret, 'c t h w -> t c h w', t=self.num_frames)
        return self.resize_centercrop(ret)

    def zoom(self, vid, d):
        h, w = vid.shape[-2:]
        r = 1 - self.zoom_ratio
        target_h, target_w = h * r, w * r
        frames = []
        for i in range(self.num_frames):
            frame = vid[:, i]
            if d == -1: # zoom in
                curr_r = i * (r - 1) / self.num_frames + 1
            else: # zoom out
                curr_r = i * (1 - r) / self.num_frames + r

            h1 = int( h * (1 - curr_r) / 2 )
            h2 = h - h1
            w1 = int( w * (1 - curr_r) / 2 )
            w2 = w - w1
            frame = T.Resize((h, w))(frame[:, h1:h2, w1:w2])
            frames.append(frame)
        # frames = einops.rearrange(torch.stack(frames), 't c h w -> c t h w', t=self.num_frames)
        frames = torch.stack(frames)
        return self.resize_centercrop(frames)

    def rotate(self, vid, d):
        degrees = []
        if d == -1:
            degrees = [self.max_degree * i / self.num_frames - self.max_degree for i in range(self.num_frames)]
        else:
            degrees = [- self.max_degree * i / self.num_frames + self.max_degree for i in range(self.num_frames)]
        # if d == -1:
        #     degrees = [self.max_degree * i / self.num_frames - self.max_degree / 2 for i in range(self.num_frames)]
        # else:
        #     degrees = [- self.max_degree * i / self.num_frames + self.max_degree / 2 for i in range(self.num_frames)]
        vid = einops.rearrange(vid, 'c t h w -> t h w c', t=self.num_frames)
        frames = []
        for curr_degree, frame in zip(degrees, vid):
            rotated = myrotate(frame.numpy(), curr_degree)
            res = get_proper_resize_size_by_size(rotated.shape[:2], self.resolution)
            rotated = einops.rearrange(rotated, 'h w c -> c h w')

            rotated = T.Resize((res))(torch.tensor(rotated))
            rotated = T.CenterCrop(self.resolution)(rotated)


            frames.append(rotated)

        # frames = einops.rearrange(frames, 't c h w -> c t h w')
        frames = torch.stack(frames)
        return self.resize_centercrop(frames)




    def __call__(self, vid, type): # vid: c t h w (torch.Tensor), type: one of  `MOTION_TYPES`
        if type == LEFT or type == RIGHT:
            return self.left_right(vid, -1 if type == LEFT else 1)
        elif type == UP or type == DOWN:
            return self.up_down(vid, -1 if type == UP else 1)
        elif type == ZOOM_IN or type == ZOOM_OUT:
            return self.zoom(vid, -1 if type == ZOOM_IN else 1)
        elif type == ROTATE_LEFT or type == ROTATE_RIGHT:
            return self.rotate(vid, -1 if type == ROTATE_LEFT else 1)
        return vid
