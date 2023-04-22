#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dataset_loader.h"
#include "hand_modeler.h"

int main(int argc, char** argv )
{
    if (argc != 2)
    {
        std::cout << "usage: ./hand_tracking [dataset_directory]" << std::endl;
        return -1;
    }

    DatasetLoader dataset{argv[1]};
    HandModeler skin_model{24, 5};

    int frame = 0;
    cv::namedWindow("Hand Model", cv::WINDOW_AUTOSIZE);
    while (!dataset.is_done()) {
        std::pair<cv::Mat, cv::Mat> image_pair = dataset.get_image_pair();

        auto hand_model_start = std::chrono::high_resolution_clock::now();
        cv::Mat hand_model = skin_model.estimate_hand(image_pair.first);
        if (hand_model.empty()) {
            frame++;
            continue;
        }
        auto hand_model_end = std::chrono::high_resolution_clock::now();

        auto hand_model_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(hand_model_end - hand_model_start);
        std::cout << "Hand model: " << hand_model_duration.count() << " ns" << std::endl;

        cv::imshow("Hand Model", hand_model);
        cv::setWindowTitle("Hand Model", "Frame " + std::to_string(frame));
        cv::waitKey(42);

        frame++;
    }
    cv::destroyAllWindows();

    return 0;
}
