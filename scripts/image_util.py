import cv2
import numpy as np

def local_contrast_normalization(x):
    # First perform global contrast normalization
    g_mean = np.mean(x)
    g_std = np.std(x)
    if g_std == 0:
        g_std = 1
    gcn_x = (x - g_mean) / g_std
    
    # Next perform local contrast
    l_mean_x = cv2.GaussianBlur(gcn_x, (9, 9), 1)
    l_mean_x2 = cv2.GaussianBlur(gcn_x * gcn_x, (9, 9), 1)
    
    lcn_x = (gcn_x - l_mean_x) / np.sqrt(np.abs(l_mean_x - l_mean_x2) + 0.001)
    return lcn_x
    
def generate_heatmap_gt(joint_uvs):
    gaussian_1d = cv2.getGaussianKernel(18, 0.1 * 18)
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    c = np.max(gaussian_2d)
    normalized = gaussian_2d / c
    
    heatmaps = np.zeros((21, 18, 18))
    for j in range(21):
        joint = joint_uvs[j]
        translation = np.array([
            [1, 0, joint[0].item() - 9],
            [0, 1, joint[1].item() - 9]
        ])
        heatmaps[j] = cv2.warpAffine(normalized, translation, (18, 18))
    return heatmaps