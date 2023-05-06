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

    bool visualize = true;
    int frame = 0;
    cv::namedWindow("Depth Image");
    cv::namedWindow("Heatmap");

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
                    cv::waitKey(0);
                }

                // Project on XY, YZ, and XZ planes
                projector.load_data(depth, bbox, gt);
                auto projections = projector.get_projections();
                auto heatmap_uvs = projector.get_heatmap_uvs();
                cv::Mat xy = projections[0];

                if (visualize) {
                    for (int i = 0; i < 3; i++) {
                        auto& plane = projections[i];
                        auto& heatmap = heatmap_uvs[i];

                        cv::Mat recolor;
                        cv::cvtColor(plane, recolor, CV_GRAY2RGB);
                        for (int j = 0; j < 21; j++) {
                            cv::circle(recolor, heatmap[j], 1, cv::Scalar(0,0, 255, 0), CV_FILLED, CV_AA, 0);
                        }

                        cv::Size resize(4 * plane.cols, 4 * plane.rows);
                        cv::Mat resized_image;
                        cv::resize(recolor, resized_image, resize);
                        cv::resizeWindow("Heatmap", resize);
                        cv::imshow("Heatmap", resized_image);
                        cv::waitKey(0);
                    }
                }

                frame++;
            }
        }
    }

    cv::destroyAllWindows();

    return 0;
}
