import os
import argparse

import numpy as np

from vot.dataset import load_dataset
from vot.region.io import write_trajectory, read_trajectory
from vot.region import Special, Rectangle
from vot.region import RegionType


EXPERIMENT_TYPE = 'baseline'

def move_results(dataset_path, src_dir, dst_dir):
    dataset = load_dataset(dataset_path)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for seq_idx, sequence in enumerate(dataset):
        
        print('Processing sequence: %s (%d/%d)' % (sequence.name, seq_idx + 1, len(dataset)))

        dst_dir_path = os.path.join(dst_dir, EXPERIMENT_TYPE, sequence.name)
        dst_file_path = os.path.join(dst_dir_path, '%s_001.txt' % sequence.name)

        if not os.path.exists(dst_dir_path):
            os.makedirs(dst_dir_path)
        
        src_path = os.path.join(src_dir, sequence.name, '%s.bin' % sequence.name)
        if not os.path.exists(src_path):
            src_path = os.path.join(src_dir, sequence.name, '%s.txt' % sequence.name)
            if not os.path.exists(src_path):
                print('Error: This path does not exist:')
                print(src_path)
                exit(-1)
        trajectory = read_trajectory(src_path)

        trajectory_out = []
        for i, prediction in enumerate(trajectory):
            if i == 0:
                trajectory_out.append(Special(1))
            else:
                if prediction.type == RegionType.MASK:
                    if prediction.is_empty():
                        out_region = Rectangle(0, 0, 0, 0)
                    else:
                        out_region = prediction.convert(RegionType.RECTANGLE)
                elif prediction.type == RegionType.RECTANGLE:
                    out_region = prediction
                else:
                    print('Error: This region type is not supported yet:', prediction.type)
                    print('Frame index:', i)
                    print('Prediction:', prediction)
                    exit(-1)
                trajectory_out.append(out_region)
        
        if len(trajectory) != len(trajectory_out):
            print('Error: Input and output trajectory lengths do not match (%d != %d)' % (len(trajectory), len(trajectory_out)))
        
        write_trajectory(dst_file_path, trajectory_out)
    
    print('Done.')

def main():
    parser = argparse.ArgumentParser(description='Move DiDi results into the workspace.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path.')
    parser.add_argument('--src', type=str, required=True, help='Source directory path.')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory path.')

    args = parser.parse_args()

    move_results(args.dataset_path, args.src, args.dst)

if __name__ == "__main__":
    main()
