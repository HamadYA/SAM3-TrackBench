import sys
import os
root = os.getcwd()
sys.path.append(root)

import numpy as np
from PIL import Image

from dam4sam_tracker import DAM4SAMTracker
import torch

import utils.vot_helper as vot

import random
import yaml
import gc, copy
from utils.mask_utils import mask2box

import warnings
warnings.filterwarnings('ignore')


# ================================================================================

def mask2box(mask):
    if mask is None:
        return None
    mask_bin = mask > 0
    if mask_bin.sum() == 0:
        return None

    x_idxs = np.where(mask_bin.sum(0) > 0)[0]
    y_idxs = np.where(mask_bin.sum(1) > 0)[0]

    x0 = x_idxs.min()
    x1 = x_idxs.max()
    y0 = y_idxs.min()
    y1 = y_idxs.max()

    bbox = [x0, y0, x1 - x0, y1 - y0]
    return bbox

# ================================================================================

with open(f"{root}/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def make_full_size(x, output_sz):
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
def main():

    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()

    tracker_name = "sam3"

    trackers = []
    for _ in range(len(objects)):
        tracker = DAM4SAMTracker(tracker_name=tracker_name)

        trackers.append(tracker)
        
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile)

    init_masks = [make_full_size(m, (image.width, image.height)) for m in objects]
    init_boxes = []
    for m in init_masks:
        box = mask2box(m)  # [x0, y0, x1, y1] or None
        if box is None:
            # fallback: whole-image box if an empty mask sneaks in (rare)
            box = [0, 0, image.width - 1, image.height - 1]
        # clamp and cast to int
        x0, y0, x1, y1 = box
        x0 = int(max(0, min(x0, image.width - 1)))
        y0 = int(max(0, min(y0, image.height - 1)))
        x1 = int(max(0, min(x1, image.width - 1)))
        y1 = int(max(0, min(y1, image.height - 1)))
        # ensure proper ordering if needed
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        init_boxes.append([x0, y0, x1, y1])

    _ = [tracker.initialize(image, None, box)
         for tracker, box in zip(trackers, init_boxes)]

    while True:
        imagefile = handle.frame()
        print("Imagefile", imagefile)
        if not imagefile:
            break

        image = Image.open(imagefile)

        outputs_states = [tracker.track(image) for tracker in trackers]
        statuses = [outputs['pred_mask'] for outputs in outputs_states]

        handle.report(statuses)

if __name__ == "__main__":
    main()

