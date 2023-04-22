#include "dataset_loader.h"

DatasetLoader::DatasetLoader(const std::string &dataset_dir)
{
    m_index = 0;

    std::string substring = "BB_left";

    // TO-DO: load images in order with an association file
    for (const auto &file : std::filesystem::directory_iterator(dataset_dir)) {
        std::string image_path = file.path();
        if (image_path.find(substring) != std::string::npos) {
            m_image_paths.push_back(image_path);
        }
    }
}

std::pair<cv::Mat, cv::Mat> DatasetLoader::get_image_pair()
{
    std::string left_image = m_image_paths[m_index];
    std::string right_image = std::regex_replace(left_image, std::regex("left"), "right");
    
    m_index++;

    return std::make_pair(cv::imread(left_image, cv::IMREAD_COLOR), cv::imread(right_image, cv::IMREAD_COLOR));
}

bool DatasetLoader::is_done()
{
    return m_index >= m_image_paths.size();
}