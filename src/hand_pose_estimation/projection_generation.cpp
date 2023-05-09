#include <chrono>
#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "dataset_loader.h"
#include "depth_projector.h"

#define SHOW_TIME

const std::string planes[3] = {"XY", "YZ", "ZX"};

int main(int argc, char** argv )
{
    if (argc != 3)
    {
        std::cout << "usage: ./projection_generation [dataset_directory] [output_directory]" << std::endl;
        return -1;
    }

    CVPR17_MSRAHandGesture dataset{argv[1]};
    DepthProjector projector(CVPR17_MSRAHandGesture::width, CVPR17_MSRAHandGesture::height, 
                             CVPR17_MSRAHandGesture::focal_length,
                             96, 18);
    const std::string output_directory{argv[2]};

    bool visualize = false;
    bool visualize_joints = false;
    if (visualize) {
        cv::namedWindow("Depth Image");
        cv::namedWindow("Heatmap 0");
        cv::namedWindow("Heatmap 1");
        cv::namedWindow("Heatmap 2");
    }

    for (int subject = 0; subject < 9; subject++) {
        for (int gesture = 0; gesture < 17; gesture++) {
            int frame = 0;
            int total_time_ns = 0;

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
                    cv::resize(depth, resized_depth, resize, 0, 0, cv::INTER_NEAREST);
                    resized_depth /= 1000.0;
                    cv::resizeWindow("Depth Image", resize);
                    cv::imshow("Depth Image", resized_depth);
                }

                // Project on XY, YZ, and XZ planes
                auto projection_start = std::chrono::high_resolution_clock::now();
                projector.load_data(depth, bbox, gt);
                auto projection_end = std::chrono::high_resolution_clock::now();
            
                // total time 
                auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(projection_end - projection_start);
            #ifdef SHOW_TIME
                std::cout << "Frame: " << frame << " " << total_duration.count() << " ns" << std::endl;
            #endif
                total_time_ns += total_duration.count();

                auto projections = projector.get_projections();
                auto heatmap_uvs = projector.get_heatmap_uvs();
                auto bboxes = projector.get_proj_bbox();
                auto ks = projector.get_proj_k();

                if (visualize) {
                    for (int i = 0; i < 3; i++) {
                        auto& plane = projections[i];
                        auto& heatmap = heatmap_uvs[i];

                        cv::Mat recolor;
                        cv::cvtColor(plane, recolor, CV_GRAY2RGB);
                        cv::Size resize(4 * plane.cols, 4 * plane.rows);
                        cv::Mat resized_image;
                        cv::resize(recolor, resized_image, resize, 0, 0, cv::INTER_NEAREST);

                        if (visualize_joints) {
                            for (int j = 0; j < 21; j++) {
                                cv::circle(resized_image, 4 * heatmap[j], 5, cv::Scalar(0,0, 255, 0), CV_FILLED, CV_AA, 0);
                            }
                        }

                        cv::resizeWindow("Heatmap " + std::to_string(i), resize);
                        cv::imshow("Heatmap " + std::to_string(i), resized_image);
                    }
                }

                // Write out data for training
                const std::string& subject_name = dataset.get_subject_name(subject);
                const std::string& gesture_name = dataset.get_gesture_name(gesture); 

                // Reformat the index to be 6 digits
                std::stringstream ss;
                ss << std::setw(6) << std::setfill('0') << frame;
                const std::string index_name{ss.str()};

                // Create the directory if it doesn't exist
                std::string current_directory = output_directory + "/" + subject_name + "/" + gesture_name;
                std::filesystem::create_directories(current_directory);
                std::string path = current_directory + "/" + index_name;

                // The projections, parameters, and heatmap labels all depend on the projection plane
                for (int i = 0; i < 3; i++) {
                    std::string projection_path = path + "_" + planes[i];
                    std::ofstream plane(projection_path + "-projection.txt");
                    for (int y = 0; y < 96; y++) {
                        for (int x = 0; x < 96; x++) {
                            plane << projections[i].at<float>(y, x) << " ";
                        }
                        plane << std::endl;
                    }
                    plane.close();

                    std::ofstream params(projection_path + "-params.txt");
                    const auto& bbox = bboxes[i];
                    params << bbox.x << " " << bbox.y << " " << ks[i] << " " << bbox.width << " " << bbox.height << std::endl;
                    for (int j = 0; j < 21; j++) {
                        const cv::Point2f& joint = heatmap_uvs[i][j];
                        float scale = 18.0 / 96.0;
                        params << (int) (scale * joint.x) << " " << (int) (scale * joint.y) << std::endl;
                    }
                    params.close();
                }

                // Also write out the relative transformation and bbox length of the point cloud
                std::ofstream common(path + "-common.txt");
                common << projector.get_x_length() << " " << projector.get_y_length() << " " << projector.get_z_length() << std::endl;
                const Eigen::Matrix4f& relative_xform = projector.get_relative_xform();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        common << relative_xform(i, j) << " ";
                    }
                    common << std::endl;
                }
                common.close(); 

                // Wait for user input before continuing
                if (visualize) {
                    cv::waitKey(0);
                }

                frame++;
            }

            // Only evaluate time for the first subject and gesture
        #ifdef SHOW_TIME
            std::cout << "Average time: " << total_time_ns / (float) frame << std::endl;
            break;
        #endif
        }
    
        // Only evaluate time for the first subject and gesture
    #ifdef SHOW_TIME
        break;
    #endif
    }

    cv::destroyAllWindows();

    return 0;
}
