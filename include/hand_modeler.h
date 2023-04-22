#ifndef __HAND_MODELER_H__
#define __HAND_MODELER_H__

#include <iostream>
#include <opencv2/video.hpp>

typedef cv::Point3_<uint8_t> Pixel;

class HandModeler
{
private:
    int m_wait_frames, m_frame_count;
    cv::Ptr<cv::BackgroundSubtractor> m_adaptive_gmm;

    cv::Mat m_hand_hist, m_image_hist;

public:
    HandModeler(int fps, int wait_time);

    cv::Mat estimate_hand(const cv::Mat &image);
};

#endif // __HAND_MODELER_H__