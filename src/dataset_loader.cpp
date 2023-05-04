#include "dataset_loader.h"

// ICIP17 Stereo Hand Pose Tracking
ICIP17_StereoHandPose::ICIP17_StereoHandPose(const std::string &dataset_dir)
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

std::pair<cv::Mat, cv::Mat> ICIP17_StereoHandPose::get_image_pair()
{
    std::string left_image = m_image_paths[m_index];
    std::string right_image = std::regex_replace(left_image, std::regex("left"), "right");
    
    m_index++;

    return std::make_pair(cv::imread(left_image, cv::IMREAD_COLOR), cv::imread(right_image, cv::IMREAD_COLOR));
}

bool ICIP17_StereoHandPose::is_done()
{
    return m_index >= m_image_paths.size();
}

// CVPR 2017 MSRA Hand Pose Estimation
CVPR17_MSRAHandGesture::CVPR17_MSRAHandGesture(const std::string &dataset_dir)
{
    m_dataset_dir = dataset_dir;
}  

void CVPR17_MSRAHandGesture::set_current_subject(int subject, int gesture)
{
    const std::string subject_names[9] = {"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"};
    const std::string gesture_names[17] = {"1", "2", "3", "4", "5", "6", "7", "8", "9",
                                           "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"};
    
    // Check if the provided subjects and gestures are valid
    assert(subject >= 0 && subject < 9 && gesture >= 0 && gesture < 17);
    std::string subject_name = subject_names[subject];
    std::string gesture_name = gesture_names[gesture];

    // If everything checks out, we can start loading the joint.txt file
    // to get the ground truth joints.
    // TODO: load ground truth joints
}

std::pair<cv::Mat, CVPR17_MSRAHandGesture::HandPose> CVPR17_MSRAHandGesture::get_next_image()
{
    // TODO: get the latest images
}
