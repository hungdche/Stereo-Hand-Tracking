#include "hand_segmentation.h"

float HandSegmenter::compute_avg_depth() {
    if (prev_segment.empty()) return -1;
    float depth = 0, count = 0;

    prev_segment.forEach<Pixel>([&](Pixel &pixel, const int position[]) -> void {
        float curr_depth = prev_segment.at<float>(position[0], position[1]);
        if (curr_depth > 0) { count++; depth += curr_depth; }
    });

    return depth / count;
}

cv::Mat HandSegmenter::compute_p_hand(const cv::Mat& p_skin, const cv::Mat& curr_frame) {
    float avg_depth = compute_avg_depth();
    
    cv::Mat mask = cv::Mat::zeros(curr_frame.rows, curr_frame.cols, CV_32F);
    if (avg_depth == -1) {
        mask.forEach<Pixel>([&](Pixel &pixel, const int position[]) -> void {
            float & current_pixel = mask.at<float>(position[0], position[1]);
            current_pixel = p_skin.at<float>(position[0], position[1]) > 0.4;
        });
    } else {
        // initialize random generator
        std::random_device rd; 
        std::mt19937 gen(rd()); 

        mask.forEach<Pixel>([&](Pixel &pixel, const int position[]) -> void {
            uint8_t & curr_pixel = mask.at<uint8_t>(position[0], position[1]);
            float curr_depth = prev_segment.at<uint8_t>(position[0], position[1]);

            Gaussian d(avg_depth, STANDARD_DEV); float noise = d(gen);
            curr_pixel = (curr_depth * noise) > 0.1;
        });
    }
    double min, max;
    cv::minMaxLoc(curr_frame, &min, &max);
    std::cout << "DEPTH " << avg_depth << " MIN " << min << " MAX " << max << std::endl;
    prev_segment = mask.mul(curr_frame);
    return prev_segment;
    
}
