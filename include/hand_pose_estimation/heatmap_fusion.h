#ifndef __HEATMAP_FUSION__
#define __HEATMAP_FUSION__

#include "depth_projector.h"
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

class HeatmapFuser {
private:
    // Set at initialization (parameters of the camera)
    int m_x_res, m_y_res;
    double m_focal_length;
    int m_out_size, m_heat_size;

    std::array<double, 3> bbox_x;
    std::array<double, 3> bbox_y;
    std::array<double, 3> bbox_w;
    std::array<double, 3> bbox_h;

    DepthProjector * depth_projector;

    std::vector<std::vector<Eigen::Vector4f> > pca_eigen_vecs_bb;	// PCA_SZ x HEAT_NUM
	std::vector<Eigen::Vector4f> pca_means_bb;		// HEAT_NUM

	std::vector<Eigen::Vector4f> joints_means_bb;		// HEAT_NUM
	std::vector<Eigen::Vector4f> joints_variance_bb;	// HEAT_NUM
	std::vector<Eigen::Matrix3f> joints_covariance_bb;	// HEAT_NUM
	std::vector<Eigen::Matrix3f> joints_inv_covariance_bb;	// HEAT_NUM
    
public:
    HeatmapFuser(int x_res, int y_res, double focal_length, int out_size, int m_heat_size);

    bool load_model(const std::string &path);
    torch::Tensor load_tensor(const std::string &path);
    
    cv::PCA fuser_pca;
	int PCA_SZ;

	double bounding_box_x[3];
	double bounding_box_y[3];
	double proj_k[3];
	double bounding_box_width[3];
	double bounding_box_height[3];


	std::vector<cv::Mat> heatmaps_vec;

	int estimate_gauss_failed_cnt;

    void fuse(float* estimate_xyz);			// return xyz in world cs (96 x 96 3d space) - gauss covariance + PCA
	void fuse_sub(float* estimate_xyz);		// return xyz in world cs (96 x 96 3d space) - mean-shift

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

	void convert_PCA_wld_to_BB();	// convert PCA in world cs to BB cs

	// estimate the mean and variance (in 18 x 18 3d space) of the gaussian distribution for each joint point
	bool estimate_gauss_mean_covariance(int joint_i, Eigen::Vector4f& mean_18, Eigen::Matrix3f& covariance_18);	// get covariance matrix

	float fuse_confidence(float conf_xy, float conf_yz, float conf_zx);
};


#endif // __HEATMAP_FUSION__
