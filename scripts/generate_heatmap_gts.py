import numpy as np
import torch
import os
import io
import cv2
import sys
import shutil
from pathlib import Path 
from image_util import generate_heatmap_gt

projections_dir = Path(sys.argv[1])
heatmap_dir = projections_dir.parent.absolute() / "heatmap_gts"

def load_heatmap(file, data_dir, output_dir):
    path = os.path.join(data_dir, file)

    bbox = np.zeros(5)
    joint_uvs = np.zeros((21, 2))

    with open(path, 'r') as f:
        line = f.readline().split(' ')
        for i in range(5):
            bbox[i] = float(line[i])
        lines = f.readlines()
        for j in range(21):
            line = lines[j].split(' ')
            joint_uvs[j][0] = float(line[0])
            joint_uvs[j][1] = float(line[1])
    heatmap_gts = generate_heatmap_gt(joint_uvs)

    for j in range(21):
        joint = str(j).zfill(2)
        output = os.path.join(output_dir, file.replace("params", "heatmap-" + joint))
        heatmap_gt = heatmap_gts[j]
        print("Writing to", output)
        with open(output, 'w') as f:
            for y in range(heatmap_gt.shape[0]):
                line = ""
                for x in range(heatmap_gt.shape[1]):
                    line += str(heatmap_gt[y, x]) + " "
                line += "\n"
                f.write(line)

for subject in sorted(os.listdir(projections_dir)):
    subject_dir = projections_dir / subject
    for gesture in sorted(os.listdir(subject_dir)):
        data_dir = subject_dir / gesture
        output_dir = heatmap_dir / subject / gesture

        if os.path.exists(output_dir):
            print(f'{output_dir} already exists')
        else:
            os.makedirs(output_dir)

        for file in sorted(os.listdir(data_dir)):
            if "-params.txt" in file:
                load_heatmap(file, data_dir, output_dir)
    
    # Only run on subject P0
    break