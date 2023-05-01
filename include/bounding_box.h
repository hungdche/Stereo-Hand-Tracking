#ifndef __BOUNDING_BOX_H__
#define __BOUNDING_BOX_H__

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#define BOX_VERTEX_NUM 8
#define BOX_EDGE_NUM 12

#define JOINT_NUM 21

class BoundingBox
{
private:
    std::vector<Eigen::Matrix4f> m_xyz_data, m_gt_xyz_data;
    cv::Rect m_projected_bbox[3];
    float m_proj_k[3];

public:
    BoundingBox();

    double x_length, y_length, z_length;
    Eigen::Matrix4f relative_xform;

	bool create_obb(float* xyz_data, int data_num, float* GT_xyz_data, int GT_data_num);
	bool project_direct(cv::Mat* proj_im, cv::Point2f proj_uv[][JOINT_NUM], int sz);
    
    cv::Rect* get_proj_bbox() {return m_projected_bbox;}
    float* get_proj_k() {return m_proj_k;}
    double get_z_value(double x_val, double y_val) {return 0.0f;}
    const std::vector<Eigen::Matrix4f>& get_gt_xyz_data() {return m_gt_xyz_data;}
    void get_project_points(float* xyz_data, int data_num, float* xy_data, float* yz_data, float* zx_data);
	Eigen::Vector3f get_yaw_pitch_roll();

private:
    void jacobbi(const Eigen::Matrix3f &input_mat, Eigen::Matrix3f &v, double *p_array);
};

#endif