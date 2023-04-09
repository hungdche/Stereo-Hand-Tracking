import numpy as np
import cv2, cv2.ximgproc

class StereoMatcher:
    def __init__(self, num_disparities, block_size, alpha, beta):
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.alpha = alpha
        self.beta = beta

        self.left_matcher = cv2.StereoSGBM_create(0, num_disparities, block_size)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.disparity_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)

        self.disparity_filter.setLambda(8000.0)
        self.disparity_filter.setSigmaColor(1.5)

    def compute_disp_maps(self, left_im, right_im):
        left_disp = self.left_matcher.compute(left_im, right_im)
        right_disp = self.right_matcher.compute(right_im, left_im)
        return self.disparity_filter.filter(left_disp, left_im, disparity_map_right=right_disp)

    def estimate_foreground(self, image):
        return self.foreground_estimator.apply(image)
