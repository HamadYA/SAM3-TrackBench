import sys
root = os.getcwd()
sys.path.append(root)

import numpy as np
from PIL import Image

from tracker import DAM4SAMTracker
import torch

import utils.vot_helper as vot

import random
import os
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


class Args:
    def __init__(self):
        self.max_num_indices = 30 # Number of candidate previous frames
        self.selection_strategy = "s2l_pos_feat_v2", # choices=["none", "s2l_pos_feat_v2"], help="Frame selection strategy")
        self.bias_mode = "kalman_pos" # choices=["none", "kalman_pos",], help="Mode of Cross Attention Bias")
        self.bias_type = "v3" # choices=["v3", "none"], help="Type of Cross Attention Bias")
        self.use_prior_prompt =False
        self.alpha = 0.3 # help="Fusion weight in frame selection."

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


def clone_tracker(base_tracker: DAM4SAMTracker) -> DAM4SAMTracker:
    """
    Create a lightweight clone of DAM4SAMTracker that reuses the base tracker's
    model weights to avoid loading multiple copies of the network.
    """
    clone = object.__new__(DAM4SAMTracker)
    clone.tracker_name = base_tracker.tracker_name
    clone.tracking_times = []
    clone._needs_preflight = False
    clone.input_image_size = base_tracker.input_image_size
    clone.img_mean = base_tracker.img_mean
    clone.img_std = base_tracker.img_std
    clone.predictor = base_tracker.predictor  # share model weights
    clone.tracker_type = base_tracker.tracker_type
    clone.args = Args()
    return clone

@torch.inference_mode()
def main():

    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()

    tracker_name = "sam3"
    # Load a single model and reuse it across per-object tracker wrappers
    shared_tracker = DAM4SAMTracker(tracker_name=tracker_name)
    trackers = [shared_tracker] + [clone_tracker(shared_tracker) for _ in range(len(objects) - 1)]

        
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile)

    init_masks = [make_full_size(m, (image.width, image.height)) for m in objects]
    init_boxes = []
    for m in init_masks:
        box = mask2box(m)
        if box is None:
            box = [0, 0, image.width - 1, image.height - 1]
        x0, y0, x1, y1 = box
        x0 = int(max(0, min(x0, image.width - 1)))
        y0 = int(max(0, min(y0, image.height - 1)))
        x1 = int(max(0, min(x1, image.width - 1)))
        y1 = int(max(0, min(y1, image.height - 1)))
        if x1 < x0: x0, x1 = x1, x0
        if y1 < y0: y0, y1 = y1, y0
        init_boxes.append([x0, y0, x1, y1])

    for init_box, tracker in zip(init_boxes, trackers):
        tracker.initialize(image, None, init_box)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        image = Image.open(imagefile)

        statuses = []
        for tracker in trackers:
            out = tracker.track(image)
            statuses.append(out['pred_mask'] if isinstance(out, dict) else out)

        handle.report(statuses)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
