import cv2
import numpy as np
import scipy.io
from tqdm.notebook import tqdm
import os, zipfile

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_zip(self, sequence):
        labels_path = os.path.join(self.data_dir, '%sCounting_BB.mat'% sequence)
        img_zip_path = os.path.join(self.data_dir, '%sCounting.zip'% sequence)
        images_path = os.path.join(self.data_dir, '%s'% sequence)

        if not os.path.isfile(img_zip_path):
            print(f'Sequence {sequence} is not found')
            return

        if not os.path.isfile(labels_path):
            print(f'Sequence {sequence} does not have labels')
            return

        left, right = [], []
        label = scipy.io.loadmat(labels_path)['handPara']
        with zipfile.ZipFile(img_zip_path, 'r') as imgpath:
            for img_name in tqdm(imgpath.infolist(), desc='Extracting %s'%sequence):
                if 'left' in img_name.filename:
                    data = imgpath.read(img_name)
                    left.append(cv2.imdecode(np.frombuffer(data, np.uint8), 3))
                if 'right' in img_name.filename:
                    data = imgpath.read(img_name)
                    right.append(cv2.imdecode(np.frombuffer(data, np.uint8), 3))
        return (left, right, label)
