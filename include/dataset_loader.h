#ifndef __DATASET_LOADER_H__
#define __DATASET_LOADER_H__

#include <chrono>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

class DatasetLoader 
{
private:
    int m_index;
    std::vector<std::string> m_image_paths;

public:
    DatasetLoader(const std::string &dataset_dir);

    std::pair<cv::Mat, cv::Mat> get_image_pair();

    bool is_done();
};

#endif // __DATASET_LOADER_H__