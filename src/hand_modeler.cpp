#include "hand_modeler.h"

HandModeler::HandModeler(int fps, int wait_time) :
    m_frame_count{0},
    m_fps{fps},
    m_wait_time{wait_time},
    m_adaptive_gmm{cv::createBackgroundSubtractorMOG2()}
{
}

cv::Mat HandModeler::estimate_hand(const cv::Mat &image)
{
    m_frame_count++;

    cv::Mat foreground;
    m_adaptive_gmm->apply(image, foreground);

    // Calculate color histogram
    cv::calcHist(&image, 1, channels, cv::Mat(), m_image_hist, 3, hist_size, ranges, true, true);
    cv::calcHist(&image, 1, channels, foreground, m_hand_hist, 3, hist_size, ranges, true, true);

    return image;
}