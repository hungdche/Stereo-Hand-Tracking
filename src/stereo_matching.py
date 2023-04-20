import numpy as np
import cv2, cv2.ximgproc, cv2.stereo

class StereoMatcher:
    def __init__(self, num_disparities, block_size = 3, alpha = 2, beta = 0.5):
        # params
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.alpha = alpha
        self.beta = beta

        self.left_matcher = cv2.StereoBM_create(num_disparities, block_size)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
    
    def compute_disp(self, left, right):
        block_offset = self.block_size // 2
        rows, cols = left.shape
        
        left = self.census_transform(left)
        right = self.census_transform(right)
        disp = np.ones((rows, cols), dtype=np.uint8)*1000
        sec_disp = np.ones((rows, cols), dtype=np.uint8)*1000

        for r in range(0, rows):
            for c in range(0, cols):
                best_dis, second_best_dis = 1000,1000
                vec1 = left[r][c]
                for c_off in range(block_offset, c):
                    vec2 = right[r][c_off]
                    ham_dis = self.hamming_distance(vec1, vec2)
                    if ham_dis <= best_dis:
                        second_best_dis = best_dis
                        best_dis = ham_dis
                    second_best_dis = min(ham_dis, second_best_dis)
                disp[r][c] = best_dis
                sec_disp[r][c] = second_best_dis
            print(r, c)
        return disp, sec_disp                
    
    def hamming_distance(self, num1, num2):
        return (num1 ^ num2).bit_count()


    def census_transform(self, image):
        '''
        taken from this: https://stackoverflow.com/a/38269363
        '''
        block_offset = self.block_size // 2
        
        rows, cols = image.shape
        image_bordered = cv2.copyMakeBorder(image, block_offset, block_offset, block_offset, block_offset, 
                                   cv2.BORDER_CONSTANT, 0)
        census = np.zeros((rows, cols), dtype=np.uint32)

        offsets = [(row, col) for row in range(self.block_size) for col in range(self.block_size) if not row == block_offset + 1 == col]
        for (row, col) in offsets:
            census = (census << 1) | (image_bordered[row:row + rows, col:col + cols] >= image)
        return census