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

    std::array<cv::Mat, 3> m_projections;
    std::array<std::array<cv::Point2f, 21>, 3> m_joint_uvs;

    // Helper functions
    bool load_depth_image(const cv::Mat &depth_image, const cv::Rect &bbox);
    void load_ground_truth(const std::array<Eigen::Vector3f, 21> &gt);

    void create_obb();

    void create_projections();
    void create_heatmap_uvs();

public:
    DepthProjector(int x_res, int y_res, double focal_length, int out_size, int m_heat_size);

    bool load_data(const cv::Mat &depth_image, const cv::Rect &bbox, const std::array<Eigen::Vector3f, 21> &gt);

    const std::array<cv::Mat, 3>& get_projections() const {return m_projections;}
    const std::array<std::array<cv::Point2f, 21>, 3>& get_heatmap_uvs() const {return m_joint_uvs;}
    const std::array<cv::Rect, 3>& get_proj_bbox() const {return m_projected_bbox;}
    const std::array<float, 3>& get_proj_k() const {return m_proj_k;}
    const Eigen::Matrix4f& get_relative_xform() const {return m_relative_xform;}
    double get_x_length() const {return m_x_length;}
    double get_y_length() const {return m_y_length;}
    double get_z_length() const {return m_z_length;}

private:
    void jacobbi(const Eigen::Matrix3f &input_mat, Eigen::Matrix3f &v, double *p_array);
};

#endif