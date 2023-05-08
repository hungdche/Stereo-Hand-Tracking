#include <chrono>
#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
// #include <torch/torch.h>
// #include <torch/script.h>

#include "dataset_loader.h"
#include "depth_projector.h"
#include "heatmap_fusion.h"

// const std::string planes[3] = {"XY", "YZ", "ZX"};

int main(int argc, char** argv )
{
    if (argc != 4)
    {
        std::cout << "usage: ./fusion_visualization [projections_directory] [heatmaps_dir] [pca_dir]" << std::endl;
        return -1;
    }

    CVPR17_MSRAHandGesture dataset{argv[1]};
    GTHeatmapLoader heatmap_loader{argv[2]};
    HeatmapFuser fuser{CVPR17_MSRAHandGesture::width, CVPR17_MSRAHandGesture::height, 
                       CVPR17_MSRAHandGesture::focal_length,
                       96, 18};

    fuser.load_pca(argv[3]);

    bool visualize = true;
    if (visualize) {
        cv::namedWindow("Heatmap 0");
        cv::namedWindow("Heatmap 1");
        cv::namedWindow("Heatmap 2");
    }

    // We only evaluate the P0 
    for (int subject = 0; subject < 9; subject++) {
        for (int gesture = 0; gesture < 17; gesture++) {
            int image_index = 0;

            dataset.set_current_set(subject, gesture);
            heatmap_loader.set_current_set(subject, gesture);
            while (!dataset.is_done()) {
                auto data = dataset.get_next_image();
                cv::Mat depth = std::get<0>(data);
                cv::Rect bbox = std::get<1>(data);
                CVPR17_MSRAHandGesture::HandPose gt = std::get<2>(data);

                auto heatmaps = heatmap_loader.get_next_heatmaps();
                
                if (visualize) {
                    for (int i = 0; i < 3; i++) {
                        auto heatmap = heatmaps[0][i];

                        cv::Size resize(20 * heatmap.cols, 20 * heatmap.rows);
                        cv::Mat resized_heatmap;
                        cv::resize(heatmap, resized_heatmap, resize, 0, 0, cv::INTER_NEAREST);

                        cv::resizeWindow("Heatmap " + std::to_string(i), resize);
                        cv::imshow("Heatmap " + std::to_string(i), resized_heatmap);
                    }
                }

                fuser.load_data(depth, bbox, gt);
                fuser.load_heatmaps(heatmaps);

                fuser.fuse();

                // Wait for user input before continuing
                if (visualize) {
                    cv::waitKey(0);
                }

                image_index++;
                break;
            }

            break;
        }

        break;
    }

    cv::destroyAllWindows();

    return 0;

    return 0;
}
