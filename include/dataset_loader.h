#ifndef __DATASET_LOADER_H__
#define __DATASET_LOADER_H__

#include <chrono>
#include <filesystem>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

class ICIP17_StereoHandPose 
{
private:
    int m_index;
    std::vector<std::string> m_image_paths;

public:
    ICIP17_StereoHandPose(const std::string &dataset_dir);

    std::pair<cv::Mat, cv::Mat> get_image_pair();

    bool is_done();
};

class CVPR17_MSRAHandGesture
{
public:
    static const int num_joints = 21;
    typedef std::array<Eigen::Vector3f, num_joints> HandPose;

private:
    std::string m_dataset_dir;
    std::string m_current_subject, m_current_gesture;

    std::vector<HandPose> m_current_gts;

public:
    CVPR17_MSRAHandGesture(const std::string &dataset_dir);

    void set_current_subject(int subject, int gesture);
    std::pair<cv::Mat, HandPose> get_next_image();
};

#endif // __DATASET_LOADER_H__