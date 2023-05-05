#ifndef __BOUNDING_BOX_H__
#define __BOUNDING_BOX_H__

#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#define BOX_VERTEX_NUM 8
#define BOX_EDGE_NUM 12

#define JOINT_NUM 21

class DepthProjector
{
private:
    // Set at initialization (parameters of the camera)
    int m_x_res, m_y_res;
    double m_focal_length;
    int m_out_size, m_heat_size;

    // Computed for every depth image
    std::vector<Eigen::Vector4f> m_xyz_data, m_gt_xyz_data;
    std::array<cv::Rect, 3> m_projected_bbox;
    std::array<float, 3> m_proj_k;
    Eigen::Vector3f m_center_pt;
    double m_x_length, m_y_length, m_z_length;
    Eigen::Matrix4f m_relative_xform;

public:
    DepthProjector(int x_res, int y_res, double focal_length, int out_size, int m_heat_size);

    bool load(const cv::Mat &depth_image, const cv::Rect &bbox, const std::array<Eigen::Vector3f, 21> &gt);
    bool load_depth_image(const cv::Mat &depth_image, const cv::Rect &bbox);
    void load_ground_truth(const std::array<Eigen::Vector3f, 21> &gt);

    bool create_obb();

    std::array<cv::Mat, 3> create_projections();
    std::array<Eigen::Matrix<double, 14, 2>, 3> create_heatmaps();

    std::array<cv::Mat, 3> project(const cv::Mat &depth_image, const cv::Rect &bbox, cv::Point2f projected_uvs[][JOINT_NUM]);
	bool project_direct(cv::Mat* proj_im, cv::Point2f proj_uv[][JOINT_NUM], int sz);
    
    std::array<cv::Rect, 3>& get_proj_bbox() {return m_projected_bbox;}
    std::array<float, 3>& get_proj_k() {return m_proj_k;}
    double get_z_value(double x_val, double y_val) {return 0.0f;}
    const std::vector<Eigen::Vector4f>& get_gt_xyz_data() {return m_gt_xyz_data;}
    void get_project_points(float* xyz_data, int data_num, float* xy_data, float* yz_data, float* zx_data);
	Eigen::Vector3f get_yaw_pitch_roll();

private:
    void jacobbi(const Eigen::Matrix3f &input_mat, Eigen::Matrix3f &v, double *p_array);
};

#endif