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
    HandModeler model(24, 5);

    cv::namedWindow("Hand Model", cv::WINDOW_AUTOSIZE);
    while (!dataset.is_done()) {
        std::pair<cv::Mat, cv::Mat> image_pair = dataset.get_image_pair();
        cv::Mat hand_model = model.estimate_hand(image_pair.first);

        cv::imshow("Hand Model", hand_model);
        cv::waitKey(42);
    }
    cv::destroyAllWindows();

    return 0;
}
