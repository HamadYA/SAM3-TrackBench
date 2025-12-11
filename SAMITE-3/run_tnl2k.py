import os
import yaml
import torch
import random
import argparse
import numpy as np

from utils.mask_utils import mask2box, save_boxes
from utils.dataset_utils import get_dataset, pil2array
from utils.visualization_utils_edited import VisualizerTracking
from tracker import DAM4SAMTracker

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

gt_path = config["tnl2k_dataset_path"]

def load_lasot_gt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    fid = 0
    for line in gt:
        x, y, w, h = map(float, line.split(','))
        prompts[fid] = ((x, y, w, h), 0)
        fid += 1
    return prompts


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main(tracker_name, dataset_name, output_dir, selected_sequence=None, save_dir=None):

    dataset = get_dataset(dataset_name, init_masks='sam2')
    with open(os.path.join(gt_path, 'list.txt'), 'r') as f:
        sequences = [line_.strip() for line_ in f.readlines()]

    for i, sequence_name in enumerate(sequences):

        if output_dir is None:
            visualizer = VisualizerTracking(save_dir)
        
        if selected_sequence is not None and selected_sequence != sequence_name:
            continue

        groundtruth_path = os.path.join(gt_path, sequence_name, 'groundtruth.txt')
        if not os.path.exists(groundtruth_path):
            continue

        if output_dir is not None:
            output_path = os.path.join(output_dir, '%s.txt' % sequence_name)
            if os.path.exists(output_path):
                print(f'{sequence_name} has already been processed. Skipping...')
                continue        

        tracker = DAM4SAMTracker()
        
        seq_len = dataset.get_seq_len(sequence_name)
        predictions = []

        print(sequence_name)

        for frame_idx in range(seq_len):
            img = dataset.get_pil_frame(sequence_name, frame_idx)

            if frame_idx == 0:
                prompts = load_lasot_gt(groundtruth_path)
                pred_bbox, track_label = prompts[0]
                _ = tracker.initialize(img, init_mask=None, bbox=pred_bbox)
                pred_mask = None

            else:
                outputs = tracker.track(img)

                pred_mask = outputs['pred_mask']
                pred_bbox = mask2box(pred_mask)

            if pred_bbox is None:
                predictions.append([0, 0, 0, 0])

            else:
                predictions.append(pred_bbox)
            
            if output_dir is None:
                visualizer.show(pil2array(img), box=pred_bbox)
        
        if output_dir is not None:
            output_path = os.path.join(output_dir, '%s.txt' % sequence_name)
            save_boxes(output_path, predictions)
            print('Results saved to:', output_path)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, default=None, help='got | lasot | lasot_ext ')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory.')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')

    args = parser.parse_args()

    args.dataset_name = 'tnl2k'
    dataset_name = args.dataset_name
    tracker_name = 'sam3'
    
    args.output_dir = 'out'
    args.save_dir = "runs"

    if args.output_dir is not None:
        base_output_dir = os.path.join(args.output_dir, tracker_name)
        run_idx = 0
        output_dir = os.path.join(base_output_dir, dataset_name, '%03d' % run_idx)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = None

    main(tracker_name, dataset_name, output_dir=output_dir, selected_sequence=args.sequence, save_dir=args.save_dir)



