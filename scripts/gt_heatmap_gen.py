from PIL import Image
import numpy as np
import torch
import os
import io
import cv2
import shutil
import click
from image_util import generate_heatmap_gt

def load_heat_map(path, output_dir, to_tensor=False):
    '''
    path are expected to be 00000_XY-projection.txt 
    '''
    bbox = np.zeros(5)
    joint_uvs = np.zeros((21, 2))

    frame = path[path.rfind('/')+1:path.find('-')]
    plane = frame[frame.find('_')+1:]
    frame = frame[:frame.find('_')]

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

    if to_tensor:
        heatmap_gt = torch.from_numpy(heatmap_gt)
        output_file = os.path.join(output_dir, '{}.pt'.format(frame))
        f = io.BytesIO()
        torch.save(heatmap_gt, output_file, _use_new_zipfile_serialization=True)
        with open(output_file, "wb") as out_f:
            out_f.write(f.getbuffer())
    else:
        output_frame_dir = os.path.join(output_dir, frame)
        output_frame_dir = os.path.join(output_frame_dir, plane)
        os.makedirs(output_frame_dir)
        for i in range(heatmap_gt.shape[0]):
            idx = str(i) if i >= 10 else '0' + str(i)
            output_file = os.path.join(output_frame_dir, '{idx}.jpg'.format(idx=idx))
            norm_image = cv2.normalize(heatmap_gt[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            norm_image = norm_image.astype(np.uint8)
            cv2.imwrite(output_file, norm_image)


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