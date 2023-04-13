"""Data preprocessing."""

import os
import pickle
import cv2
from typing import Any, Tuple

import numpy as np
import scipy.io
from tqdm.notebook import tqdm

def unzip_dataset(seq='train'):
    import zipfile
    

    labels_path = os.path.join('data', '%sCounting_BB.mat'% seq)
    img_zip_path = os.path.join('data', '%sCounting.zip'% seq)
    images_path = os.path.join('data', '%s'% seq)

    if not os.path.isdir(img_zip_path):
        print(f'Sequence {seq} is not found')
        return 
    
    if not os.path.isfile(labels_path):
        print(f'Sequence {seq} does not have labels') 
        return
    
    left = []
    right = []
    with zipfile.ZipFile(img_zip_path, 'r') as imgpath:
        for member in tqdm(imgpath.infolist(), desc='Extracting %s'%seq):
            imgpath.extract(member, images_path)

def get_data(seq):
    print(f'Loading data for sequence {seq}...')
    labels_path = os.path.join('data', '%sCounting_BB.mat'% seq)
    images_path = os.path.join('data', '%s'% seq)
    images = os.listdir(images_path)

    left = []
    right = []
    label = scipy.io.loadmat(labels_path)['handPara']
    for f in tqdm(images, desc='Loading %s'%seq):
        if 'left' in f:
            left.append(cv2.imread(os.path.join(images_path,f)))
        elif 'right' in f:
            left.append(cv2.imread(os.path.join(images_path,f)))

    left  = np.asarray([cv2.imread(os.path.join(images_path,f)) for f in images if 'left' in f])
    right  = np.asarray([cv2.imread(os.path.join(images_path,f)) for f in images if 'right' in f])
    # assert left.shape != right.shape
    print("Left shape:", left.shape)
    print("Right shape:", right.shape)
    return (left, right, label)

