import numpy as np
import cv2
from matplotlib import pyplot as plt

class HandModeler:
    def __init__(self, fps=24, wait_time=5):
        self.foreground_estimator = cv2.createBackgroundSubtractorMOG2()
        self.frame_count = 0
        self.fps = fps
        self.wait_time = wait_time

        self.hand_hist = np.zeros((256, 256, 256))
        self.image_hist = np.zeros((256, 256, 256))

    def adaptive_gmm(self, image):
        return self.foreground_estimator.apply(image)

    def estimate_hand(self, image):
        self.frame_count += 1
        hand = self.adaptive_gmm(image)
        self.hand_hist += cv2.calcHist([image], [0, 1, 2], hand, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        self.image_hist += cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        if (self.frame_count <= self.fps * self.wait_time):
            return None
        hand_count = self.hand_hist[(image[:, :, 0], image[:, :, 1], image[:, :, 2])]
        image_count = self.image_hist[(image[:, :, 0], image[:, :, 1], image[:, :, 2])]
        P_skin = np.where(image_count > 0, hand_count / image_count, 0)
        return P_skin
