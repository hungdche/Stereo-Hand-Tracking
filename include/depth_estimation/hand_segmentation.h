#ifndef __HAND_SEGMENTATION_H__
#define __HAND_SEGMENTATION_H__

#include <iostream>
#include <random>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stereo/stereo.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#define STANDARD_DEV 0.15

typedef std::normal_distribution<float> Gaussian;
typedef cv::Point3_<uint8_t> Pixel;

class HandSegmenter {
private:
    cv::Mat prev_segment;
    float compute_avg_depth();
    
public:
    HandSegmenter(){};

    cv::Mat compute_p_hand(const cv::Mat& p_skin, const cv::Mat& curr_frame);      
};

#endif // __HAND_SEGMENTATION_H__