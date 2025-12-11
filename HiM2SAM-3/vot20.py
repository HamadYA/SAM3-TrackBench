import sys

import numpy as np
from PIL import Image

from tracker import DAM4SAMTracker
import torch

import utils.vot_helper as vot

import random
import os
import yaml

root = os.getcwd()
with open(f"{root}/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

def get_vot_mask(masks_list, image_width, image_height):
    id_ = 1
    masks_multi = np.zeros((image_height, image_width), dtype=np.float32)
    for mask in masks_list:
        m = make_full_size(mask, (image_width, image_height))
        masks_multi[m>0] = id_
        id_ += 1
    return masks_multi


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    tracker = DAM4SAMTracker(tracker_name="sam3")

    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()
    
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile)

    init_masks = [make_full_size(m, (image.width, image.height)) for m in objects]
    m = tracker.initialize(image, init_masks)

    while True:
        imagefile = handle.frame()
        print("Imagefile", imagefile)
        if not imagefile:
            break

        image = Image.open(imagefile)

        outputs = tracker.track(image)
        
        status = [outputs['pred_mask']]# [m for m in outputs['pred_mask']]
        handle.report(status)

if __name__ == "__main__":
    main()
