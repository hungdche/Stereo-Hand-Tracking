#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dataset_loader.h"
#include "hand_modeler.h"
#include "stereo_matching.h"
#include "hand_segmentation.h"

#define STEREO_DEBUG
// #define SHOW_TIME

int main(int argc, char** argv )
{
    if (argc != 2)
    {
        std::cout << "usage: ./hand_tracking [dataset_directory]" << std::endl;
        return -1;
    }

    ICIP17_StereoHandPose dataset{argv[1]};
    HandModeler skin_model{24, 5};
    StereoMatcher stereo_matcher{16, 5};
    HandSegmenter hand_segmenter{};

    int frame = 0;
    cv::namedWindow("Segmented Hand", cv::WINDOW_AUTOSIZE);
    while (!dataset.is_done()) {
        std::pair<cv::Mat, cv::Mat> image_pair = dataset.get_image_pair();
        
        /* HAND MODELING */
        auto hand_model_start = std::chrono::high_resolution_clock::now();
        cv::Mat hand_model = skin_model.estimate_hand(image_pair.first);
        if (hand_model.empty()) {
            frame++;
            continue;
        }
        auto hand_model_end = std::chrono::high_resolution_clock::now();

        /* STEREO MATCHING */
        auto stereo_matching_start = std::chrono::high_resolution_clock::now();
        cv::Mat matched_img;
    #ifdef STEREO_DEBUG
        cv::imshow("Hand Model", hand_model);
        matched_img = stereo_matcher.debug(image_pair.first, image_pair.second, hand_model);
    #else
        matched_img = stereo_matcher.compute(image_pair.first, image_pair.second, hand_model);
    #endif
        auto stereo_matching_end = std::chrono::high_resolution_clock::now();

        /* HAND SEGMENTATION */
        auto hand_segmentation_start = std::chrono::high_resolution_clock::now();
        // cv::Mat segmented_hand = hand_segmenter.compute_p_hand(hand_model, matched_img);
        auto hand_segmentation_end = std::chrono::high_resolution_clock::now();

        /* PRINT COMPUTATION TIME */
        // hand model
    #ifdef SHOW_TIME
        auto hand_model_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(hand_model_end - hand_model_start);
        std::cout << "Hand model: " << hand_model_duration.count() << " ns" << std::endl;
        // stereo matching
        auto stereo_matching_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stereo_matching_end - stereo_matching_start);
        std::cout << "Stereo Matching: " << stereo_matching_duration.count() << " ns" << std::endl;
        // hand segmentation
        auto hand_segmentation_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(hand_segmentation_end - hand_segmentation_start);
        std::cout << "Hand Segmentation: " << hand_segmentation_duration.count() << " ns" << std::endl;
        // total time 
        auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(hand_segmentation_end - hand_model_start);
        std::cout << "Total Time: " << total_duration.count() << " ns" << std::endl;
    #endif

        /* PRINT IMAGES */
        cv::imshow("Segmented Hand", matched_img);
        cv::setWindowTitle("Segmented Hand", "Frame " + std::to_string(frame));
        cv::waitKey(42);

        frame++;
    }
    cv::destroyAllWindows();

    return 0;
}
