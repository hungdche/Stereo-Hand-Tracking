from PIL import Image
import numpy as np
import torch
import os
import shutil
import click
from image_util import generate_heatmap_gt

def load_heat_map(path,output_dir):
    bbox = np.zeros(5)
    joint_uvs = np.zeros((21, 2))

    frame = path[path.rfind('/')+1:path.find('-')]
    
    with open(path) as f:
        line = f.readline().split(' ')
        for i in range(5):
            bbox[i] = float(line[i])
        lines = f.readlines()
        for j in range(21):
            line = lines[j].split(' ')
            joint_uvs[j][0] = float(line[0])
            joint_uvs[j][1] = float(line[1])
    heatmap_gt = generate_heatmap_gt(joint_uvs)
    heatmap_gt = torch.from_numpy(heatmap_gt)
    torch.save(heatmap_gt, os.path.join(output_dir, '{}.pt'.format(frame)))


if __name__ == "__main__":

    @click.command(no_args_is_help=True)
    @click.argument('folder', nargs=-1)
    def main(folder):
        """ 
        python gt_heatmap_gen.py <folder> # follow by more folder if needed
        <folder> is expected to be P0 to P8
        """
        for f in folder:
            parent_dir = os.path.abspath(os.path.join(f, os.pardir))
            output_dir = os.path.join(parent_dir, os.path.basename(os.path.normpath(f))+"_gt")

            if os.path.exists(output_dir):
                print(f'{output_dir} already exists')
                shutil.rmtree(output_dir)

            os.makedirs(output_dir)
            
            for sub_dir in os.listdir(f):
                current_dir = os.path.join(f, sub_dir)
                sub_parent_dir = os.path.join(output_dir, sub_dir)
                os.makedirs(sub_parent_dir)
                for file in sorted(os.listdir(current_dir)):
                    if "-params.txt" in file:
                        load_heat_map(os.path.join(current_dir, file), sub_parent_dir)
                print(f'Finish converting {os.path.join(current_dir, file)} to heatmaps')
    main()