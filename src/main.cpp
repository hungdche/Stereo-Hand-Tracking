#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dataset_loader.h"
#include "hand_modeler.h"
#include "stereo_matching.h"

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

    int frame = 0, debugged = 0;
    cv::namedWindow("Hand Model", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Left Hand", cv::WINDOW_AUTOSIZE);
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
        cv::imshow("Hand Model", hand_model);
        if (!debugged) {
            debugged = 1;
            stereo_matcher.debug(image_pair.first, image_pair.second, hand_model);
        } 
        std::cout << debugged << std::endl;
        auto stereo_matching_end = std::chrono::high_resolution_clock::now();

        /* PRINT COMPUTATION TIME */
        auto hand_model_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(hand_model_end - hand_model_start);
        std::cout << "Hand model: " << hand_model_duration.count() << " ns" << std::endl;
        auto stereo_matching_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stereo_matching_end - stereo_matching_start);
        std::cout << "Stereo Matching: " << stereo_matching_duration.count() << " ns" << std::endl;

        /* PRINT IMAGES */
        cv::imshow("Hand Model", hand_model);
        cv::setWindowTitle("Hand Model", "Frame " + std::to_string(frame));
        cv::waitKey(42);

        frame++;
    }
    cv::destroyAllWindows();

    return 0;
}
