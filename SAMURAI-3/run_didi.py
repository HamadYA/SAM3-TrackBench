import os
import time
import yaml
import torch
import hydra
import random
import argparse
import numpy as np
from PIL import Image

from vot.dataset import load_dataset
from vot.region.io import read_trajectory, write_trajectory
from vot.region.shapes import Mask

from utils.visualization_utils_edited import VisualizerTracking
from tracker import DAM4SAMTracker

from utils.utils import get_seq_names
from utils.dataset_utils import pil2array
from utils.mask_utils import save_boxes

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def mask2box(mask):
    if mask is None:
        return None
    mask_bin = mask > 0
    if mask_bin.sum() == 0:
        return None
    x_idxs = np.where(mask_bin.sum(0)>0)[0]
    y_idxs = np.where(mask_bin.sum(1)>0)[0]
    x0 = x_idxs.min()
    x1 = x_idxs.max()
    y0 = y_idxs.min()
    y1 = y_idxs.max()
    bbox = [x0, y0, x1, y1]
    return bbox


def mask2box_original(mask):
    if mask is None:
        return None
    mask_bin = mask > 0
    if mask_bin.sum() == 0:
        return None
    x_idxs = np.where(mask_bin.sum(0)>0)[0]
    y_idxs = np.where(mask_bin.sum(1)>0)[0]
    x0 = x_idxs.min()
    x1 = x_idxs.max()
    y0 = y_idxs.min()
    y1 = y_idxs.max()
    bbox = [x0, y0, x1-x0+1, y1-y0+1]
    return bbox

@torch.inference_mode()
@torch.cuda.amp.autocast()
def run_sequence(tracker_name, dataset_path, sequence_names, output_dir=None, save_dir=None):
    
    dataset = load_dataset(dataset_path)

    for sequence_name in sequence_names:

        if output_dir is not None:
            if os.path.exists(os.path.join(output_dir, sequence_name, f"{sequence_name}.txt")):
                continue

        if output_dir is not None:
            output_path = os.path.join(output_dir, '%s.txt' % sequence_name)
            if os.path.exists(output_path):
                print(f'{sequence_name} has already been processed. Skipping...')
                continue
            
        if output_dir is None:
            visualizer = VisualizerTracking(save_dir)

        sequence = dataset[sequence_name]
        frame_idxs = list(range(len(sequence)))

        tracker = DAM4SAMTracker(tracker_name)        
        pred_masks = []
        predictions = []

        init_img = Image.open(sequence.frame(0).filename())
        img_width, img_height = init_img.width, init_img.height
        init_mask_path = os.path.join(sequence.metadata('root'), 'first_frame_segm.txt')
        mask = read_trajectory(init_mask_path)[0].rasterize((0, 0, img_width-1, img_height-1))
        
        pred_masks.append(Mask(mask))
        pred_bbox = mask2box(mask)
        outputs = tracker.initialize(init_img, mask, pred_bbox)
        pred_bbox = mask2box_original(mask)

        if pred_bbox is None:
            predictions.append([0, 0, 0, 0])
        else:
            predictions.append(pred_bbox)
        
        for i, frame_idx in enumerate(frame_idxs[1:]):
            image = Image.open(sequence.frame(frame_idx).filename())

            outputs = tracker.track(image)

            pred_mask = outputs['pred_mask']
            pred_masks.append(Mask(pred_mask))

            pred_bbox = mask2box(pred_mask)

            if output_dir is None:
                visualizer.show(pil2array(image), mask=pred_mask)

            if pred_bbox is None:
                predictions.append([0, 0, 0, 0])
            else:
                predictions.append(pred_bbox)

        if output_dir is not None:
            os.makedirs(os.path.join(output_dir, sequence.name), exist_ok=True)
            write_trajectory(os.path.join(output_dir, sequence.name, f"{sequence.name}.txt"), pred_masks)
        
        if output_dir is not None:
            save_boxes(output_path, predictions)
            print('Results saved to:', output_path)

    hydra.core.global_hydra.GlobalHydra.instance().clear()

def main():
    parser = argparse.ArgumentParser(description='Visualize sequence.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')
    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')

    args = parser.parse_args()

    args.dataset_path = config["didi_dataset_path"]
    tracker_name = f'sam3'
    args.output_dir = 'out/didi/'
    # args.save_dir = "runs"
    args.save_dir = None

    
    if args.sequence is None:
        seq_names = get_seq_names(args.dataset_path)
    else:
        seq_names = [args.sequence]

    run_sequence(tracker_name, args.dataset_path, seq_names, output_dir=args.output_dir, save_dir=args.save_dir)
    
if __name__ == "__main__":
    main()



