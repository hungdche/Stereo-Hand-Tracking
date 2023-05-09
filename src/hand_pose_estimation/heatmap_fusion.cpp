#include "heatmap_fusion.h"

HeatmapFuser::HeatmapFuser(int x_res, int y_res, double focal_length, int out_size, int heat_size) :
	m_depth_projector{x_res, y_res, focal_length, out_size, heat_size}
{
    m_x_res = x_res;
	m_y_res = y_res;
	m_focal_length = focal_length;
	m_out_size = out_size;
	m_heat_size = heat_size;

	m_pca_size = 30;

	m_pca_eigenvector_bbox.resize(max_pca_size);
	for (int i = 0; i < max_pca_size; i++)
	{
		m_pca_eigenvector_bbox[i].resize(JOINT_NUM);
	}
	m_pca_mean_bbox.resize(JOINT_NUM);

	m_joints_means_bb.resize(JOINT_NUM);
	m_joints_variance_bb.resize(JOINT_NUM);
	m_joints_covariance_bb.resize(JOINT_NUM);
	m_joints_inv_covariance_bb.resize(JOINT_NUM);

	m_num_estimate_gauss_failed = 0;
}

bool HeatmapFuser::load_pca(const std::string &path)
{
	std::ifstream eigenvectors(path + "/pca_eigenvectors.txt");
	if (!eigenvectors.is_open()) {
		return false;
	}

	m_pca_fuser.eigenvectors = cv::Mat::zeros(63, 63, CV_32FC1);
	for (int y = 0; y < 63; y++) {
		for (int x = 0; x < 63; x++) {
			eigenvectors >> m_pca_fuser.eigenvectors.at<float>(y, x);
		}
	}
	eigenvectors.close();
	
	std::ifstream eigenvalues(path + "/pca_eigenvalues.txt");
	if (!eigenvalues.is_open()) {
		return false;
	}

	m_pca_fuser.eigenvalues = cv::Mat::zeros(63, 1, CV_32FC1);
	for (int y = 0; y < 63; y++) {
		eigenvalues >> m_pca_fuser.eigenvalues.at<float>(y, 0);
	}
	eigenvalues.close();

	std::ifstream mean(path + "/pca_mean.txt");
	if (!mean.is_open()) {
		return false;
	}

	m_pca_fuser.mean = cv::Mat::zeros(1, 63, CV_32FC1);
	for (int x = 0; x < 63; x++) {
		mean >> m_pca_fuser.mean.at<float>(0, x);
	}
	mean.close();

	return true;
}

bool HeatmapFuser::load_data(const cv::Mat &depth_image, const cv::Rect &bbox, const std::array<Eigen::Vector3f, 21> &gt)
{
	if (!m_depth_projector.load_data(depth_image, bbox, gt)) {
		return false;
	}

	m_bbox = m_depth_projector.get_proj_bbox();
	m_proj_k = m_depth_projector.get_proj_k();

	for (int i = 0; i < 3; i++) {
		m_bbox_18[i].x = (int) (m_bbox[i].x * m_heat_size) / (float) m_out_size;
		m_bbox_18[i].y = (int) (m_bbox[i].y * m_heat_size) / (float) m_out_size;
		m_bbox_18[i].width = (int) (m_bbox[i].width * m_heat_size) / (float) m_out_size;
		m_bbox_18[i].height = (int) (m_bbox[i].height * m_heat_size) / (float) m_out_size;
	}
	
	return true;
}

void HeatmapFuser::load_heatmaps(std::array<std::array<cv::Mat, 3>, 21> &heatmaps)
{
	m_heatmaps = heatmaps;

	for (int j = 0; j < 21; j++) {
		for (int i = 0; i < 3; i++) {
			cv::Mat estimated_heatmap = m_heatmaps[j][i].clone();
			double min, max; 
			cv::Point min_loc, max_loc;
			cv::minMaxLoc(estimated_heatmap, &min, &max, &min_loc, &max_loc);
			estimated_heatmap /= max;
			cv::cvtColor(estimated_heatmap, m_estimated_heatmaps[j][i], cv::COLOR_GRAY2RGB);
		}
	}
}

void HeatmapFuser::fuse()	// joint optimization using covariance
{
	// 1. calculate mean and variance for each joint point
	convert_pca_world_to_bbox();

	double k = pow(double(m_out_size) / double(m_heat_size), 2);
	int joint_idx = 0;
	for (joint_idx = 0; joint_idx < JOINT_NUM; joint_idx++) {
		Eigen::Vector4f mean_18;
		if (!estimate_gauss_mean_covariance(joint_idx, mean_18, m_joints_covariance_bb[joint_idx])) {
			break;
		}

		xyz_18_96(mean_18, m_joints_means_bb[joint_idx]);
		
		m_joints_covariance_bb[joint_idx] *= k;	// covariance_18 in 18 x 18 3d space ---> covariance_96 in 96 x 96 3d space
		m_joints_inv_covariance_bb[joint_idx] = m_joints_covariance_bb[joint_idx].inverse();
	}

	if (joint_idx < JOINT_NUM) {
		m_num_estimate_gauss_failed++;
		fuse_sub();
		return;
	}

	// 2. A*alpha = b
	cv::Mat A_mat(m_pca_size, m_pca_size, CV_32FC1);
	for (int i = 0; i < m_pca_size; i++) {
		for (int j = 0; j <= i; j++) {
			float a_ij = 0.0;

			for (int k = 0; k < JOINT_NUM; k++) {
				Eigen::Vector3f tmp = m_joints_inv_covariance_bb[k] * m_pca_eigenvector_bbox[i][k].head<3>();
				a_ij += m_pca_eigenvector_bbox[j][k].head<3>().dot(tmp);
			}

			A_mat.at<float>(i, j) = a_ij;
			A_mat.at<float>(j, i) = a_ij;
		}
	}

	cv::Mat b_mat(m_pca_size, 1, CV_32FC1);
	for (int i = 0; i < m_pca_size; i++) {
		float b_i = 0.0;

		for (int k = 0; k < JOINT_NUM; k++) {
			Eigen::Vector3f tmp = m_joints_inv_covariance_bb[k] * m_pca_eigenvector_bbox[i][k].head<3>();
			Eigen::Vector3f tmp_diff;

			for (int ki = 0; ki < 3; ki++) {
				tmp_diff[ki] = m_joints_means_bb[k][ki] - m_pca_mean_bbox[k][ki];
			}

			b_i += tmp_diff.dot(tmp);
		}

		b_mat.at<float>(i, 0) = b_i;
	}

	cv::Mat alpha_mat = A_mat.inv() * b_mat;

	// 3. recover from PCA
	Eigen::Vector3f offset_vec(0.0, 0.0, 0.0);//(1.6, -1.2, -1.2);	//
	for (int joint_idx = 0; joint_idx < JOINT_NUM; joint_idx++) {
		Eigen::Vector4f estimate_pt(m_pca_mean_bbox[joint_idx]);
		for (int i_pca = 0; i_pca < m_pca_size; i_pca++) {
			for (int i_xyz = 0; i_xyz < 3; i_xyz++) {
				estimate_pt[i_xyz] += alpha_mat.at<float>(i_pca, 0) * m_pca_eigenvector_bbox[i_pca][joint_idx][i_xyz];
			}
		}

		//*
		//////////////draw estimate points on heat-maps//////////////
		Eigen::Vector4f estimate_pt_18;
		xyz_96_18(estimate_pt, estimate_pt_18);
		int u, v;
		_3d_xy(estimate_pt_18[0], estimate_pt_18[1], u, v);
		if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size) {
			std::cout << "Joint " << joint_idx << ": xy out" << std::endl;
		}
		cv::circle(m_estimated_heatmaps[joint_idx][0], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		_3d_yz(estimate_pt_18[1], estimate_pt_18[2], u, v);
		if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size) {
			std::cout << "Joint " << joint_idx << ": yz out" << std::endl;
		}
		cv::circle(m_estimated_heatmaps[joint_idx][1], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		_3d_zx(estimate_pt_18[2], estimate_pt_18[0], u, v);
		if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size) {
			std::cout << "Joint " << joint_idx << ": zx out" << std::endl;
		}
		cv::circle(m_estimated_heatmaps[joint_idx][2], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		//////////////draw estimate points on heat-maps//////////////
		//*/

		estimate_pt = m_depth_projector.get_relative_xform() * estimate_pt; // convert from BB cs to world cs
		m_estimated_joints_xyz[joint_idx] = estimate_pt.head<3>() - offset_vec;
	}
}

void HeatmapFuser::fuse_sub()	// respectively optimization
{
	Eigen::Vector3f offset_vec(0, 0, 0); //(-3.26, 4.24, -0.32);	//(-0.46716, -0.27824, -1.5545);	//(1.6, -1.2, -1.2);	//
	for (int i = 0; i < JOINT_NUM; i++)
	{
		Eigen::Vector4f estimate_pt = estimate_joint_xyz(i);
		estimate_pt = m_depth_projector.get_relative_xform() * estimate_pt;
		m_estimated_joints_xyz[i] = estimate_pt.head<3>() - offset_vec;
	}
}

Eigen::Vector4f HeatmapFuser::estimate_joint_xyz(int joint)
{
	double h = 10;	// 50
	double lambda[3] = {3, 1, 1};

	double h_2 = pow(h, 2);
	double threshold = 1e-5;

	Eigen::Vector4f cur_pt(0.0, 0.0, 0.0, 1.0);	// = get_start_point(joint_i);	// 
	
	Eigen::Vector4f pre_pt, diff;

	cv::Mat xy_8bit(m_heat_size, m_heat_size, CV_8UC1);
	cv::Mat yz_8bit(m_heat_size, m_heat_size, CV_8UC1);
	cv::Mat zx_8bit(m_heat_size, m_heat_size, CV_8UC1);
	for (int v = 0; v < m_heat_size; v++) {
		for (int u = 0; u < m_heat_size; u++) {
			xy_8bit.at<unsigned char>(v, u) = (int) (m_heatmaps[joint][0].at<float>(v, u) * 256);
			yz_8bit.at<unsigned char>(v, u) = (int) (m_heatmaps[joint][1].at<float>(v, u) * 256);
			zx_8bit.at<unsigned char>(v, u) = (int) (m_heatmaps[joint][2].at<float>(v, u) * 256);
		}
	}

	std::array<cv::Mat, 3> heat_binary;
	cv::threshold(xy_8bit, heat_binary[0], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(yz_8bit, heat_binary[1], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(zx_8bit, heat_binary[2], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	std::vector<cv::Point3d> pts_weights[3];		// pt.x save u, pt.y save v, pt.z save weight
	for (int i = 0; i < 3; ++i) {
		for (int v = 0; v < m_heat_size; v++) {
			for (int u = 0; u < m_heat_size; u++) {
				if (heat_binary[i].at<bool>(v, u) == 255) {
					cv::Point3d pt;
					pt.z = m_heatmaps[joint][i].at<float>(v, u);
					_2d_3d(i, u, v, pt);
					pts_weights[i].push_back(pt);
				}
			}
		}
	}

	do {
		pre_pt = cur_pt;
		double x_num1 = 0.0, x_den1 = 0.0, y_num1 = 0.0, y_den1 = 0.0, z_num2 = 0.0, z_den2 = 0.0;
		double x_num3 = 0.0, x_den3 = 0.0, y_num2 = 0.0, y_den2 = 0.0, z_num3 = 0.0, z_den3 = 0.0;

		// xy
		for (int j = 0; j < pts_weights[0].size(); j++) {
			double exp_term = 
				exp(-(pow(cur_pt[0] - pts_weights[0][j].x, 2) + pow(cur_pt[1] - pts_weights[0][j].y, 2)) / h_2);
			double tmp = pts_weights[0][j].z * exp_term;
			x_den1 += tmp;
			y_den1 += tmp;
			x_num1 += tmp*pts_weights[0][j].x;
			y_num1 += tmp*pts_weights[0][j].y;
		}
		
		// yz
		for (int j = 0; j < pts_weights[1].size(); j++) {
			double exp_term =
				exp(-(pow(cur_pt[1] - pts_weights[1][j].x, 2) + pow(cur_pt[2] - pts_weights[1][j].y, 2)) / h_2);
			double tmp = pts_weights[1][j].z * exp_term;
			y_den2 += tmp;
			z_den2 += tmp;
			y_num2 += tmp*pts_weights[1][j].x;
			z_num2 += tmp*pts_weights[1][j].y;
		}

		// zx
		for (int j = 0; j < pts_weights[2].size(); ++j) {
			double exp_term =
				exp(-(pow(cur_pt[2] - pts_weights[2][j].x, 2) + pow(cur_pt[0] - pts_weights[2][j].y, 2)) / h_2);
			double tmp = pts_weights[2][j].z * exp_term;
			z_den3 += tmp;
			x_den3 += tmp;
			z_num3 += tmp*pts_weights[2][j].x;
			x_num3 += tmp*pts_weights[2][j].y;
		}

		cur_pt[0] = (lambda[0] * x_num1 + lambda[2] * x_num3) / (lambda[0] * x_den1 + lambda[2] * x_den3);
		cur_pt[1] = (lambda[0] * y_num1 + lambda[1] * y_num2) / (lambda[0] * y_den1 + lambda[1] * y_den2);
		cur_pt[2] = (lambda[1] * z_num2 + lambda[2] * z_num3) / (lambda[1] * z_den2 + lambda[2] * z_den3);
		//cur_pt[2] = (z_num2 + z_num3) / (z_den2 + z_den3);

		diff = pre_pt - cur_pt;
	} while (diff.squaredNorm() > threshold);

	//*/
	int u, v;
	_3d_xy(cur_pt[0], cur_pt[1], u, v);
	if (u<0 || v<0 || u >= m_heat_size || v >= m_heat_size) {
		std::cout << "Joint " << joint << ": xy out" << std::endl;
	}
	cv::circle(m_estimated_heatmaps[joint][0], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
	_3d_yz(cur_pt[1], cur_pt[2], u, v);
	if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size) {
		std::cout << "Joint " << joint << ": yz out" << std::endl;
	}
	cv::circle(m_estimated_heatmaps[joint][1], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
	_3d_zx(cur_pt[2], cur_pt[0], u, v);
	if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size) {
		std::cout << "Joint " << joint << ": zx out" << std::endl;
	}
	cv::circle(m_estimated_heatmaps[joint][2], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);

	Eigen::Vector4f pt_96;
	xyz_18_96(cur_pt, pt_96);

	return pt_96;
}

bool HeatmapFuser::estimate_gauss_mean_covariance(int joint, Eigen::Vector4f& mean_18, Eigen::Matrix3f& covariance_18)
{
	cv::Mat xy_8bit(m_heat_size, m_heat_size, CV_8UC1);
	cv::Mat yz_8bit(m_heat_size, m_heat_size, CV_8UC1);
	cv::Mat zx_8bit(m_heat_size, m_heat_size, CV_8UC1);
	for (int v = 0; v < m_heat_size; v++) {
		for (int u = 0; u < m_heat_size; u++) {
			xy_8bit.at<unsigned char>(v, u) = (int) (m_heatmaps[joint][0].at<float>(v, u) * 256);
			yz_8bit.at<unsigned char>(v, u) = (int) (m_heatmaps[joint][1].at<float>(v, u) * 256);
			zx_8bit.at<unsigned char>(v, u) = (int) (m_heatmaps[joint][2].at<float>(v, u) * 256);
		}
	}

	std::array<cv::Mat, 3> heat_binary;
	cv::threshold(xy_8bit, heat_binary[0], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(yz_8bit, heat_binary[1], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(zx_8bit, heat_binary[2], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// 1. get sample points
	std::vector<Eigen::Vector4f> xyz_weights;		// xyz_weights[0]->x, xyz_weights[1]->y, xyz_weights[2]->z, xyz_weights[3]->weight
	double z_value[m_heat_size];
	for (int i = 0; i < m_heat_size; ++i) {
		z_value[i] = m_depth_projector.get_z_length() * (i + 0.5) / double(m_out_size);
	}

	for (int v = 0; v < m_heat_size; v++) {
		for (int u = 0; u < m_heat_size; u++) {
			if (heat_binary[0].at<bool>(v, u) == 0) {
				continue;
			}

			cv::Point3d pt;
			pt.z = m_heatmaps[joint][0].at<float>(v, u);
			xy_3d(u, v, pt.x, pt.y);

			for (int i = 0; i < m_heat_size; i++) {
				int u_yz, v_yz;
				_3d_yz(pt.y, z_value[i], u_yz, v_yz);
				if (u_yz < 0 || u_yz >= m_heat_size || v_yz < 0 || v_yz >= m_heat_size || !heat_binary[1].at<bool>(v_yz, u_yz)) {
					continue;
				}
				float conf_yz = m_heatmaps[joint][1].at<float>(v_yz, u_yz);

				int u_zx, v_zx;
				_3d_zx(z_value[i], pt.x, u_zx, v_zx);
				if (u_zx < 0 || u_zx >= m_heat_size || v_zx < 0 || v_zx >= m_heat_size || !heat_binary[2].at<bool>(v_zx, u_zx)) {
					continue;
				}
				float conf_zx = m_heatmaps[joint][2].at<float>(v_zx, u_zx);

				xyz_weights.push_back(Eigen::Vector4f(pt.x, pt.y, z_value[i], fuse_confidence(pt.z, conf_yz, conf_zx)));
			}
		}
	}

	if (xyz_weights.size() == 0) {
		return false;
	}

	// 2. get mean
	Eigen::Vector4f xyz_sum(0.0, 0.0, 0.0, 1.0);
	float conf_sum = 0.0;

	for (int i = 0; i < xyz_weights.size(); i++)
	{
		conf_sum += xyz_weights[i][3];
		for (int k = 0; k < 3; k++) {
			xyz_sum[k] += xyz_weights[i][3] * xyz_weights[i][k];
		}
	}

	for (int i = 0; i < 3; ++i) {
		mean_18[i] = xyz_sum[i] / conf_sum;
	}
	mean_18[3] = 1.0;

	// 3. get covariance
	covariance_18 = Eigen::Matrix3f::Zero();
	Eigen::Vector4f xyz_var(0.0, 0.0, 0.0, 1.0);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j <= i; j++) {
			for (int k = 0; k < xyz_weights.size(); k++) {
				covariance_18(i, j) += xyz_weights[k][3] * (xyz_weights[k][i] - mean_18[i])
													     * (xyz_weights[k][j] - mean_18[j]);
			}
			covariance_18(i, j) /= conf_sum;
			covariance_18(j, i) = covariance_18(i, j);
		}
	}

	float epsilon = 1e-3;
	for (int i = 0; i < 3; ++i) {
		if (covariance_18(i,i) < epsilon) {
			covariance_18(i, i) = epsilon;
		}
	}

	return (covariance_18.determinant() > 1);
}

void HeatmapFuser::_2d_3d(int view_type, int u, int v, cv::Point3d& pt)
{
	if (view_type == 0) {
		xy_3d(u, v, pt.x, pt.y);
	} else if (view_type == 1) {
		yz_3d(u, v, pt.x, pt.y);
	} else if (view_type == 2) {
		zx_3d(u, v, pt.x, pt.y);
	}
}

void HeatmapFuser::xy_3d(int u, int v, double& x, double& y)
{
	x = (u - m_bbox_18[0].x + 1) / m_proj_k[0];
	y = (v - m_bbox_18[0].y + 1) / m_proj_k[0];
}

void HeatmapFuser::yz_3d(int u, int v, double& y, double& z)
{
	y = (u - m_bbox_18[1].x + 1) / m_proj_k[1];
	z = (v - m_bbox_18[1].y + 1) / m_proj_k[1];
}

void HeatmapFuser::zx_3d(int u, int v, double& z, double& x)
{
	z = (u - m_bbox_18[2].x + 1) / m_proj_k[2];
	x = (v - m_bbox_18[2].y + 1) / m_proj_k[2];
}

void HeatmapFuser::xyz_18_96(const Eigen::Vector4f& xyz_18, Eigen::Vector4f& xyz_96)
{
	xyz_96[0] = (m_out_size*xyz_18[0]) / m_heat_size - m_depth_projector.get_x_length()/ 2.0;
	xyz_96[1] = -(m_out_size*xyz_18[1]) / m_heat_size + m_depth_projector.get_y_length() / 2.0;
	xyz_96[2] = (m_out_size*xyz_18[2]) / m_heat_size - m_depth_projector.get_z_length() / 2.0;
	xyz_96[3] = 1.0;
}

void HeatmapFuser::xyz_96_18(const Eigen::Vector4f& xyz_96, Eigen::Vector4f& xyz_18)
{
	xyz_18[0] = (m_heat_size * (xyz_96[0] + m_depth_projector.get_x_length() / 2.0)) / m_out_size;
	xyz_18[1] = -(m_heat_size * (xyz_96[1] - m_depth_projector.get_y_length() / 2.0)) / m_out_size;
	xyz_18[2] = (m_heat_size * (xyz_96[2] + m_depth_projector.get_z_length() / 2.0)) / m_out_size;
	xyz_18[3] = 1.0;
}

void HeatmapFuser::_3d_xy(double x, double y, int& u, int& v)
{
	u = (int) (m_proj_k[0] * x + m_bbox_18[0].x);
	v = (int) (m_proj_k[0] * y + m_bbox_18[0].y);
}

void HeatmapFuser::_3d_yz(double y, double z, int& u, int& v)
{
	u = (int) (m_proj_k[1] * y + m_bbox_18[1].x);
	v = (int) (m_proj_k[1] * z + m_bbox_18[1].y);
}

void HeatmapFuser::_3d_zx(double z, double x, int& u, int& v)
{
	u = (int) (m_proj_k[2] * z + m_bbox_18[2].x);
	v = (int) (m_proj_k[2] * x + m_bbox_18[2].y);
}

void HeatmapFuser::convert_pca_world_to_bbox()
{
	Eigen::Matrix4f world_to_bbox = m_depth_projector.get_relative_xform().inverse();

	for (int i = 0; i < 3; i++) {
		world_to_bbox(i, 3) = 0.0;
	}

	for (int i = 0; i < m_pca_size; i++) {
		for (int j = 0; j < 21; j++) {
			Eigen::Vector4f eigen_world(0.0, 0.0, 0.0, 1.0);
			for (int k = 0; k < 3; k++) {
				eigen_world[k] = m_pca_fuser.eigenvectors.at<float>(i, j * 3 + k);
			}
			m_pca_eigenvector_bbox[i][j] = world_to_bbox * eigen_world;
		}
	}

	for (int j = 0; j < 21; j++) {
		Eigen::Vector4f mean_world(0.0, 0.0, 0.0, 1.0);
		for (int i = 0; i < 3; i++) {
			mean_world[i] = m_pca_fuser.mean.at<float>(0, j * 3 + i);
		}
		m_pca_mean_bbox[j] = world_to_bbox * mean_world;
	}
}

float HeatmapFuser::fuse_confidence(float conf_xy, float conf_yz, float conf_zx)
{
	//return pow(conf_xy, 1 / lamda[0]) * pow(conf_yz, 1 / lamda[1]) * pow(conf_zx, 1 / lamda[2]);
	return conf_xy * conf_yz * conf_zx;
}