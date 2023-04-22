#include <iostream>

#include <opencv2/opencv.hpp>

#include "dataset_loader.h"

int main(int argc, char** argv )
{
    if (argc != 2)
    {
        std::cout << "usage: ./hand_tracking [dataset_directory]" << std::endl;
        return -1;
    }

    DatasetLoader dataset{argv[1]};

    while (!dataset.is_done()) {
        auto image_pair = dataset.get_image_pair();

        cv::namedWindow("Left Image", cv::WINDOW_AUTOSIZE );
        cv::imshow("Left Image", image_pair.first);
        cv::waitKey(0);
    }

    return 0;
}
