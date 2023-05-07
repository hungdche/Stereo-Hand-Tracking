#ifndef __DATASET_LOADER_H__
#define __DATASET_LOADER_H__

#include <chrono>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

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

    static const int width = 320;
    static const int height = 240;
    static constexpr double focal_length = 241.42;

private:
    std::string m_dataset_dir;
    std::string m_current_subject, m_current_gesture;
    int m_index;

    const std::array<std::string, 9> m_subject_names;
    const std::array<std::string, 17> m_gesture_names;

    std::vector<HandPose> m_current_gts;

public:
    CVPR17_MSRAHandGesture(const std::string &dataset_dir);

    void set_current_set(int subject, int gesture);
    std::tuple<cv::Mat, cv::Rect, HandPose> get_next_image();

    const std::string& get_subject_name(int subject) const {return m_subject_names[subject];}
    const std::string& get_gesture_name(int gesture) const {return m_gesture_names[gesture];}

    bool is_done();
};

class TensorLoader {
private:
    int m_index;
    bool m_is_tensor;
    const std::string planes[3] = {"XY", "YZ", "ZX"};
    
    std::vector<std::string> m_tensor_paths;
public:
    TensorLoader(const std::string &dataset_dir, bool is_tensor=false);

    torch::Tensor get_current_tensor();
    std::vector<std::vector<cv::Mat>> get_current_images();

    bool is_done();
};

#endif // __DATASET_LOADER_H__