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
    m_index = 0;
    m_current_subject = "";
    m_current_gesture = "";
}  

void CVPR17_MSRAHandGesture::set_current_set(int subject, int gesture)
{
    const std::string subject_names[9] = {"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"};
    const std::string gesture_names[17] = {"1", "2", "3", "4", "5", "6", "7", "8", "9",
                                           "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"};
    
    // Check if the provided subjects and gestures are valid
    assert(subject >= 0 && subject < 9 && gesture >= 0 && gesture < 17);
    m_current_subject = subject_names[subject];
    m_current_gesture = gesture_names[gesture];

    // If everything checks out, we can start loading the joint.txt file
    // to get the ground truth joints.
    std::string joint_file = m_dataset_dir + "/" + m_current_subject + "/" + m_current_gesture + "/joint.txt";
    std::ifstream joints(joint_file, std::ios::in);
    if (!joints.is_open()) {
        std::cerr << "Can't open joint file at " << joint_file << std::endl;
    }
    std::string line;
    std::stringstream ss;

    // First line is the number of elements
    int num_elements;
    std::getline(joints, line);
    ss << line;
    ss >> num_elements;
    std::cout << "Reading " << num_elements << " images from subject " << m_current_subject << " with gesture " << m_current_gesture << std::endl;

    // Following lines are the x, y, z positions of the ground truth joints 
    while (std::getline(joints, line)) {
        if (!line.empty()) {
            ss << line;
            CVPR17_MSRAHandGesture::HandPose gt;
            for (int i = 0; i < num_joints; i++) {
                Eigen::Vector3f joint;
                ss >> joint.x() >> joint.y() >> joint.z();
                gt[i] = joint;
            }
            m_current_gts.push_back(gt);
        }
    }
    assert(num_elements == m_current_gts.size());
    joints.close();

    // Reset the current image index to 0
    m_index = 0;
}

std::tuple<cv::Mat, cv::Rect, CVPR17_MSRAHandGesture::HandPose> CVPR17_MSRAHandGesture::get_next_image()
{
    assert(!m_current_subject.empty() && !m_current_gesture.empty());

    // Reformat the index to be 6 digits
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << m_index;
    std::string image_name = ss.str() + "_depth.bin";
    std::string depth_file = m_dataset_dir + "/" + m_current_subject + "/" + m_current_gesture + "/" + image_name;

    // Load file
    std::ifstream depth_image(depth_file, std::ios::in | std::ios::binary);
    if (!depth_image.is_open()) {
        std::cerr << "Can't open depth image at " << depth_file << std::endl;
    }

    // First read in the first 6 dimensions
    int width, height, left, top, right, bottom;
    depth_image.read(reinterpret_cast<char*>(&width), sizeof(int));
    depth_image.read(reinterpret_cast<char*>(&height), sizeof(int));
    depth_image.read(reinterpret_cast<char*>(&left), sizeof(int));
    depth_image.read(reinterpret_cast<char*>(&top), sizeof(int));
    depth_image.read(reinterpret_cast<char*>(&right), sizeof(int));
    depth_image.read(reinterpret_cast<char*>(&bottom), sizeof(int));

    // Create a bounding box based off of the dimensions provided
    cv::Rect bbox;
    bbox.x = left;
    bbox.y = top;
    bbox.width = right - left;
    bbox.height = bottom - top;

    // Read the rest of the data in
    float *depths = new float[bbox.width * bbox.height];
    depth_image.read(reinterpret_cast<char*>(depths), sizeof(float) * bbox.width * bbox.height);
    depth_image.close();

    // Create a depth image based off of the dimensions
    cv::Mat depth(cv::Size(bbox.width, bbox.height), CV_32FC1);
    depth.forEach<float>([&](float &pixel, const int position[]) -> void {
        pixel = depths[position[0] * bbox.width + position[1]];
    });

    return std::make_tuple(depth, bbox, m_current_gts[m_index++]);
}

bool CVPR17_MSRAHandGesture::is_done()
{
    return m_index >= m_current_gts.size();
}