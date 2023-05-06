import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class ProjectionDataset(Dataset):
    def __init__(self, dataset_dir, subjects, gestures, plane):
        self.plane = plane
        self.image_prefixes = []
        for subject in subjects:
            for gesture in gestures:
                curr_dir = os.path.join(dataset_dir, subject, gesture)
                
                for path in sorted(os.listdir(curr_dir)):
                    if  "-common.txt" in path:
                        prefix = path.replace("-common.txt", "")
                        self.image_prefixes.append(os.path.join(curr_dir, prefix))
                
    
    def __len__(self):
        return len(self.image_prefixes)
    
    def __getitem__(self, idx):
        plane_path = self.image_prefixes[idx] + "_" + self.plane + "-projection.txt"
        depth = np.zeros((96, 96))
        with open(plane_path) as f:
            lines = f.readlines()
            for y in range(96):
                line = lines[y].split(' ')
                for x in range(96):
                    depth[y, x] = float(line[x])
        return depth