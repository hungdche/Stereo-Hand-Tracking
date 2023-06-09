#include <chrono>
#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
// #include <torch/torch.h>
// #include <torch/script.h>

#include "dataset_loader.h"
#include "depth_projector.h"
#include "heatmap_fusion.h"

// #define SHOW_TIME
// #define SAVE_IMAGES

// const std::string planes[3] = {"XY", "YZ", "ZX"};

int main(int argc, char** argv )
{
#ifdef SAVE_IMAGES
    if (argc != 5) {
        std::cout << "usage: ./fusion_visualization [projections_directory] [heatmaps_dir] [pca_dir]" << std::endl;
        return -1;
    }
#else
    if (argc != 4) {
        std::cout << "usage: ./fusion_visualization [projections_directory] [heatmaps_dir] [pca_dir]" << std::endl;
        return -1;
    }
#endif

    CVPR17_MSRAHandGesture dataset{argv[1]};
    GTHeatmapLoader heatmap_loader{argv[2]};
    HeatmapFuser fuser{CVPR17_MSRAHandGesture::width, CVPR17_MSRAHandGesture::height, 
                       CVPR17_MSRAHandGesture::focal_length,
                       96, 18};

    fuser.load_pca(argv[3]);

    bool visualize = true;
    bool per_joint_error = false;
    if (visualize) {
        cv::namedWindow("Heatmap 0");
        cv::namedWindow("Heatmap 1");
        cv::namedWindow("Heatmap 2");
    }

    // We only evaluate the P0 
    for (int subject = 0; subject < 9; subject++) {
        int image_count = 0;
        float average_subject_distance = 0.0f;

        for (int gesture = 0; gesture < 17; gesture++) {
            int frame = 0;
            int64_t total_time_ns = 0;
            float average_gesture_distance = 0.0f;

            dataset.set_current_set(subject, gesture);
            heatmap_loader.set_current_set(subject, gesture);
            while (!dataset.is_done()) {
                auto data = dataset.get_next_image();
                cv::Mat depth = std::get<0>(data);
                cv::Rect bbox = std::get<1>(data);
                CVPR17_MSRAHandGesture::HandPose gt_joints = std::get<2>(data);

                // Visualize the hand image:
                // resize by 4x and scale to meters
                if (visualize) {
                    cv::Size resize(4 * bbox.width, 4 * bbox.height);
                    cv::Mat resized_depth;
                    cv::resize(depth, resized_depth, resize);
                    resized_depth /= 1000.0;
                    cv::resizeWindow("Depth Image", resize);
                    cv::imshow("Depth Image", resized_depth);
                }

                auto heatmaps = heatmap_loader.get_next_heatmaps();
                
                if (visualize) {
                    for (int i = 0; i < 3; i++) {
                        auto heatmap = heatmaps[0][i];

                        double min, max; 
                        cv::Point min_loc, max_loc;
                        cv::minMaxLoc(heatmap, &min, &max, &min_loc, &max_loc);

                        cv::Size resize(20 * heatmap.cols, 20 * heatmap.rows);
                        cv::Mat resized_heatmap;
                        cv::resize(heatmap, resized_heatmap, resize, 0, 0, cv::INTER_NEAREST);
                        resized_heatmap /= max;

                        cv::resizeWindow("Heatmap " + std::to_string(i), resize);
                        cv::imshow("Heatmap " + std::to_string(i), resized_heatmap);
                    }
                }

                auto fusion_start = std::chrono::high_resolution_clock::now();
                fuser.load_data(depth, bbox, gt_joints);
                fuser.load_heatmaps(heatmaps);
                fuser.fuse();
                auto fusion_end = std::chrono::high_resolution_clock::now();

                // total time 
                auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(fusion_end - fusion_start);
            #ifdef SHOW_TIME
                std::cout << "Frame: " << frame << " " << total_duration.count() << " ns" << std::endl;
            #endif
                total_time_ns += total_duration.count();

                const auto& estimated_joints = fuser.get_estimated_joints();

                float total_distance = 0.0f;
                std::array<float, 21> distances;
                for (int i = 0; i < 21; i++) {
                    Eigen::Vector3f curr_error = estimated_joints[i] - gt_joints[i];
                    distances[i] = curr_error.norm();
                    total_distance += distances[i];

                    if (per_joint_error) {
                        std::cout << "Joint " << i << " Analysis" << std::endl;

                        std::cout << "GT: ";
                        for (int k = 0; k < 3; k++) {
                            std::cout << gt_joints[i][k] << " ";
                        }
                        std::cout << std::endl;

                        std::cout << "Estimate: ";
                        for (int k = 0; k < 3; k++) {
                            std::cout << estimated_joints[i][k] << " ";
                        }
                        std::cout << std::endl;

                        std::cout << "Error: ";
                        for (int k = 0; k < 3; k++) {
                            std::cout << curr_error[k] << " ";
                        }
                        std::cout << std::endl;

                        std::cout << "Error: " << distances[i] << std::endl;
                        std::cout << std::endl;
                    }
                }
                float average_distance = total_distance / 21;
                average_gesture_distance += average_distance;
                average_subject_distance += average_distance;

                // std::cout << "Average distance: " << average_distance << std::endl;

                const auto& estimated_heatmaps = fuser.get_estimated_heatmaps();
                if (visualize) {
                    for (int j = 0; j < 21; j++) {
                        cv::waitKey(0);
                        for (int i = 0; i < 3; i++) {
                            auto heatmap = estimated_heatmaps[j][i];

                            cv::Size resize(20 * heatmap.cols, 20 * heatmap.rows);
                            cv::Mat resized_heatmap;
                            cv::resize(heatmap, resized_heatmap, resize, 0, 0, cv::INTER_NEAREST);

                            cv::resizeWindow("Heatmap " + std::to_string(i), resize);
                            cv::imshow("Heatmap " + std::to_string(i), resized_heatmap);
                        }
                    }
                }

                // Wait for user input before continuing
                if (visualize) {
                    cv::waitKey(0);
                }

                frame++;
                image_count++;
            }

            average_gesture_distance /= frame;
            std::cout << "Average gesture error: " << average_gesture_distance << " mm" << std::endl;

            // Only evaluate time for the first subject and gesture
        #ifdef SHOW_TIME
            std::cout << "Average time: " << total_time_ns / (float) frame << std::endl;
            break;
        #endif
        }

        average_subject_distance /= image_count;
        std::cout << "Average subject error: " << average_subject_distance << " mm" << std::endl;

        // Only evaluate time for the first subject and gesture
    #ifdef SHOW_TIME
        break;
    #endif
        break;
    }

    cv::destroyAllWindows();

    return 0;

    return 0;
}
