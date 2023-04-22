#include "hand_modeler.h"

int channels[3] = {0, 1, 2};
int hist_size[3] = {256, 256, 256};
float range[2] = {0, 256};
const float *ranges[3] = {range, range, range};

HandModeler::HandModeler()
{
    m_adaptive_gmm = cv::createBackgroundSubtractorMOG2();
}

cv::Mat HandModeler::estimate_hand(const cv::Mat &image)
{
    // First estimate the foreground with adaptive GMM
    cv::Mat foreground;
    m_adaptive_gmm->apply(image, foreground);

    // Calculate color histogram of the image as a whole
    // as well as the estimated hand as the foreground
    cv::calcHist(&image, 1, channels, cv::Mat(), m_image_hist, 3, hist_size, ranges, true, true);
    cv::calcHist(&image, 1, channels, foreground, m_hand_hist, 3, hist_size, ranges, true, true);

    // Use the accumulated histograms to estimate skin probability given the pixel color
    cv::Mat P_skin = cv::Mat::zeros(image.size(), CV_32F);

    image.forEach<Pixel>([&](Pixel &pixel, const int position[]) -> void {
        int hand_count = m_hand_hist.at<float>(pixel.x, pixel.y, pixel.z);
        int image_count = m_image_hist.at<float>(pixel.x, pixel.y, pixel.z);

        P_skin.at<float>(position[0], position[1]) = (image_count > 0) ? ((float) hand_count) / image_count : 0;
    });

    return P_skin;
}