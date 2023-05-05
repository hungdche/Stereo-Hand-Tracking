#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dataset_loader.h"
#include "depth_projector.h"

int main(int argc, char** argv )
{
    if (argc != 2)
    {
        std::cout << "usage: ./projection_generation [dataset_directory]" << std::endl;
        return -1;
    }

    CVPR17_MSRAHandGesture dataset{argv[1]};
    DepthProjector projector(CVPR17_MSRAHandGesture::width, CVPR17_MSRAHandGesture::height, 
                             CVPR17_MSRAHandGesture::focal_length,
                             96, 18);

    bool visualize = false;
    int frame = 0;
    cv::namedWindow("Depth Image");

    for (int subject = 0; subject < 9; subject++) {
        for (int gesture = 0; gesture < 17; gesture++) {
            dataset.set_current_set(subject, gesture);
            while (!dataset.is_done()) {
                auto data = dataset.get_next_image();
                cv::Mat depth = std::get<0>(data);
                cv::Rect bbox = std::get<1>(data);
                CVPR17_MSRAHandGesture::HandPose gt = std::get<2>(data);

                // Visualize the hand image:
                // resize by 4x and scale to meters
                if (visualize) {
                    cv::Size resize(4 * bbox.width, 4 * bbox.height);
                    cv::Mat resized_depth;
                    cv::resize(depth, resized_depth, resize);
                    resized_depth /= 1000.0;
                    cv::resizeWindow("Depth Image", resize);
                    cv::imshow("Depth Image", resized_depth);
                    cv::waitKey(42);
                }

                // Project on XY, YZ, and XZ planes
                projector.load_data(depth, bbox, gt);
                projector.create_obb();

                auto projections = projector.create_projections();
                cv::Mat xy = projections[0];

                if (!visualize) {
                    cv::Size resize(4 * xy.cols, 4 * xy.rows);
                    cv::Mat resized_depth;
                    cv::resize(xy, resized_depth, resize);
                    cv::resizeWindow("Depth Image", resize);
                    cv::imshow("Depth Image", resized_depth);
                    cv::waitKey(42);
                }
            

                frame++;
            }
        }
    }

    cv::destroyAllWindows();

    return 0;
}
