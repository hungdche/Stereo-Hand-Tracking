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
CVPR17_MSRAHandGesture::CVPR17_MSRAHandGesture(const std::string &dataset_dir) :
    m_subject_names{"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"},
    m_gesture_names{"1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"}
{
    m_dataset_dir = dataset_dir;
    m_index = 0;
    m_current_subject = "";
    m_current_gesture = "";
}  

void CVPR17_MSRAHandGesture::set_current_set(int subject, int gesture)
{    
    // Check if the provided subjects and gestures are valid
    assert(subject >= 0 && subject < 9 && gesture >= 0 && gesture < 17);
    m_current_subject = m_subject_names[subject];
    m_current_gesture = m_gesture_names[gesture];

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
    m_current_gts.clear();
    while (std::getline(joints, line)) {
        if (!line.empty()) {
            ss << line;
            CVPR17_MSRAHandGesture::HandPose gt;
            for (int i = 0; i < num_joints; i++) {
                Eigen::Vector3f joint;
                ss >> joint.x() >> joint.y() >> joint.z();
                joint.z() *= -1.0f;
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

// GT Heatmap Loader
GTHeatmapLoader::GTHeatmapLoader(const std::string &dataset_dir) :
    m_subject_names{"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"},
    m_gesture_names{"1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "I", "IP", "L", "MP", "RP", "T", "TIP", "Y"},
    m_plane_names{"XY", "YZ", "ZX"}
{
    m_dataset_dir = dataset_dir;
    m_index = 0;
    m_current_subject = "";
    m_current_gesture = "";
}  

void GTHeatmapLoader::set_current_set(int subject, int gesture)
{
    // Check if the provided subjects and gestures are valid
    assert(subject >= 0 && subject < 9 && gesture >= 0 && gesture < 17);
    m_current_subject = m_subject_names[subject];
    m_current_gesture = m_gesture_names[gesture];

    // Find the current directory and count the number of images
    m_current_dir = m_dataset_dir + "/" + m_current_subject + "/" + m_current_gesture;

    m_num_images = 0;
    auto dir_iterator = std::filesystem::directory_iterator(m_current_dir);
    for (auto& entry : dir_iterator) {
        if (entry.is_regular_file()) {
            m_num_images++;
        }
    }
    m_num_images /= (21 * 3);

    std::cout << "Reading " << m_num_images << " heatmaps from subject " << m_current_subject << " with gesture " << m_current_gesture << std::endl;

    // Reset the current image index to 0
    m_index = 0;
}

std::array<std::array<cv::Mat, 3>, 21> GTHeatmapLoader::get_next_heatmaps()
{
    std::array<std::array<cv::Mat, 3>, 21> heatmaps;

    // Utility string stream for reformatting
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << m_index;
    std::string index_name = ss.str();
    ss.str(std::string());

    for (int j = 0; j < 21; j++) {
        // Get heatmap name
        ss << std::setw(2) << std::setfill('0') << j;
        std::string heatmap_name = "-heatmap-" + ss.str() + ".txt";
        ss.str(std::string());

        // Get all planes of the joint
        for (int i = 0; i < 3; i++) {
            std::string path = m_current_dir + "/" + index_name + "_" + m_plane_names[i] + heatmap_name;
            std::ifstream file(path);

            heatmaps[j][i] = cv::Mat::zeros(18, 18, CV_32F);
            for (int y = 0; y < 18; y++) {
                for (int x = 0; x < 18; x++) {
                    file >> heatmaps[j][i].at<float>(y, x);
                }
            }
        }
        break;
    }

    m_index++;
    return heatmaps;
}

bool GTHeatmapLoader::is_done()
{
    return m_index >= m_num_images;
}


// GT Heatmap Tensors
// TensorLoader::TensorLoader(const std::string &dataset_dir, bool is_tensor)
//     : m_is_tensor(is_tensor)
// {
//     m_index = 0;

//     std::vector<std::filesystem::path> files_in_directory;
//     std::copy(std::filesystem::directory_iterator(dataset_dir), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
//     std::sort(files_in_directory.begin(), files_in_directory.end());

//     for (const std::string &file : files_in_directory) {
//         m_tensor_paths.push_back(file);
//     }
// }

// torch::Tensor TensorLoader::get_current_tensor()
// {
//     if (!m_is_tensor) {
//         std::cout << "[TensorLoader]: path is images for get_current_tensor()" << std::endl;
//         exit(EXIT_FAILURE); 
//     }
//     std::string tensor_path = m_tensor_paths[m_index];    
//     m_index++;

//     std::ifstream input(tensor_path, std::ios::binary);
//     std::vector<char> bytes(
//         (std::istreambuf_iterator<char>(input)),
//         (std::istreambuf_iterator<char>()));
//     input.close();

//     torch::IValue x = torch::pickle_load(bytes);
//     torch::Tensor to_return = x.toTensor();

//     return to_return;
// }

// std::vector<std::vector<cv::Mat>> TensorLoader::get_current_images() {
//     if (m_is_tensor) {
//         std::cout << "[TensorLoader]: path is tensors for get_current_images()" << std::endl;
//         exit(EXIT_FAILURE); 
//     }

//     std::string current_frame = m_tensor_paths[m_index];
//     m_index++;

//     std::vector<std::vector<cv::Mat>> to_return;
//     to_return.push_back(std::vector<cv::Mat>());
//     to_return.push_back(std::vector<cv::Mat>());
//     to_return.push_back(std::vector<cv::Mat>());

//     for (int i = 0; i < 3; i++) {
//         std::string heatmap_path = current_frame + "/" + planes[i];

//         std::vector<std::filesystem::path> files_in_directory;
//         std::copy(std::filesystem::directory_iterator(heatmap_path), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
//         std::sort(files_in_directory.begin(), files_in_directory.end());

//         for (const std::string &file : files_in_directory) {
//             to_return[i].push_back(cv::imread(file, cv::IMREAD_GRAYSCALE));
//         }
//     }

//     return to_return;
// }   

// bool TensorLoader::is_done()
// {
//     return m_index >= m_tensor_paths.size();
// }