#ifndef __HAND_MODELER_H__
#define __HAND_MODELER_H__

#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

int channels[3] = {0, 1, 2};
int hist_size[3] = {256, 256, 256};
float range[2] = {0, 256};
const float *ranges[3] = {range, range, range};

class HandModeler
{
private:
    int m_frame_count, m_fps, m_wait_time;
    cv::Ptr<cv::BackgroundSubtractor> m_adaptive_gmm;

    cv::Mat m_hand_hist, m_image_hist;

public:
    HandModeler(int fps, int wait_time);

    cv::Mat estimate_hand(const cv::Mat &image);
};

#endif // __HAND_MODELER_H__