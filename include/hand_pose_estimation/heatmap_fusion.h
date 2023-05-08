#ifndef __HEATMAP_FUSION__
#define __HEATMAP_FUSION__

#include "depth_projector.h"
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
// #include <torch/torch.h>
// #include <torch/script.h>

class HeatmapFuser 
{
public:
	static const int max_pca_size = 63;

private:
    // Set at initialization (parameters of the camera)
    int m_x_res, m_y_res;
    double m_focal_length;
    int m_out_size, m_heat_size;

    std::array<cv::Rect, 3> m_bbox;
	std::array<cv::Rect, 3> m_bbox_18;
	std::array<double, 3> m_proj_k;

    DepthProjector m_depth_projector;

	int m_pca_size; 
    std::vector<std::vector<Eigen::Vector4f> > m_pca_eigenvector_bbox;	// PCA_SZ x HEAT_NUM
	std::vector<Eigen::Vector4f> m_pca_mean_bbox;		// HEAT_NUM
	cv::PCA m_pca_fuser;

	std::vector<Eigen::Vector4f> m_joints_means_bb;		// HEAT_NUM
	std::vector<Eigen::Vector4f> m_joints_variance_bb;	// HEAT_NUM
	std::vector<Eigen::Matrix3f> m_joints_covariance_bb;	// HEAT_NUM
	std::vector<Eigen::Matrix3f> m_joints_inv_covariance_bb;	// HEAT_NUM
	int m_num_estimate_gauss_failed;

	std::array<std::array<cv::Mat, 3>, 21> m_heatmaps;
	std::array<Eigen::Vector3f, 21> m_estimated_joints_xyz; 
    
public:
    HeatmapFuser(int x_res, int y_res, double focal_length, int out_size, int m_heat_size);

	bool load_pca(const std::string &path);
    bool load_data(const cv::Mat &depth_image, const cv::Rect &bbox, const std::array<Eigen::Vector3f, 21> &gt);
	bool load_heatmaps(std::array<std::array<cv::Mat, 3>, 21> &heatmaps);

    void fuse();			// return xyz in world cs (96 x 96 3d space) - gauss covariance + PCA
	void fuse_sub(float* estimate_xyz);		// return xyz in world cs (96 x 96 3d space) - mean-shift

	const std::array<Eigen::Vector3f, 21>& get_estimated_joints() const {return m_estimated_joints_xyz;}

private:
	Eigen::Vector4f estimate_joint_xyz(int joint_i);	// return xyz in BB cs (96 x 96 3d space)

	void _2d_3d(int view_type, int u, int v, cv::Point3d& pt);	// view_type: 0-xy, 1-yz, 2-zx; 18 x 18 3d space

	void xy_3d(int u, int v, double& x, double& y);	// 18 x 18 u, v ---> xyz in 18 x 18 3d space
	void yz_3d(int u, int v, double& y, double& z); // 18 x 18 u, v ---> xyz in 18 x 18 3d space
	void zx_3d(int u, int v, double& z, double& x); // 18 x 18 u, v ---> xyz in 18 x 18 3d space
	void xyz_18_96(const Eigen::Vector4f& xyz_18, Eigen::Vector4f& xyz_96);	// xyz in 18 x 18 3d space ---> xyz in 96 x 96 3d space
	void xyz_96_18(const Eigen::Vector4f& xyz_96, Eigen::Vector4f& xyz_18);	// xyz in 96 x 96 3d space ---> xyz in 18 x 18 3d space

	void _3d_xy(double x, double y, int& u, int& v); // xyz in 18 x 18 3d space ---> 18 x 18 u, v
	void _3d_yz(double y, double z, int& u, int& v); // xyz in 18 x 18 3d space ---> 18 x 18 u, v
	void _3d_zx(double z, double x, int& u, int& v); // xyz in 18 x 18 3d space ---> 18 x 18 u, v

	void convert_pca_world_to_bbox();	// convert PCA in world cs to BB cs

	// estimate the mean and variance (in 18 x 18 3d space) of the gaussian distribution for each joint point
	bool estimate_gauss_mean_covariance(int joint_i, Eigen::Vector4f& mean_18, Eigen::Matrix3f& covariance_18);	// get covariance matrix

	float fuse_confidence(float conf_xy, float conf_yz, float conf_zx);
};


#endif // __HEATMAP_FUSION__
