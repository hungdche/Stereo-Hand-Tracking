#ifndef __HAND_SEGMENTATION_H__
#define __HAND_SEGMENTATION_H__

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stereo/stereo.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

class HandSegmenter {
private:
public:
    HandSegmenter();

    void segment(const cv::Mat& p_skin);    
};



#endif // __HAND_SEGMENTATION_H__