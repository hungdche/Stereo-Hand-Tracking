#include <chrono>
#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "dataset_loader.h"
#include "depth_projector.h"
#include "heatmap_fusion.h"

// const std::string planes[3] = {"XY", "YZ", "ZX"};

int main(int argc, char** argv )
{
    if (argc != 2)
    {
        std::cout << "usage: ./fusion_visualization [tensors_directory]" << std::endl;
        return -1;
    }

    TensorLoader tensors{argv[1], false};
    HeatmapFuser fuser(CVPR17_MSRAHandGesture::width, CVPR17_MSRAHandGesture::height, 
                       CVPR17_MSRAHandGesture::focal_length,
                       96, 18);
    
    while (!tensors.is_done()) {
        std::vector<std::vector<cv::Mat>> a = tensors.get_current_images();
    }

    return 0;
}
