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

void HeatmapFuser::fuse(float* estimate_xyz)	// joint optimization using covariance
{
	for (int i = 0; i < 3; i++)
	{
		bbox_x[i] = (bounding_box_x[i] * m_heat_size) / m_out_size;
		bbox_y[i] = (bounding_box_y[i] * m_heat_size) / m_out_size;
		bbox_w[i] = (bounding_box_width[i] * m_heat_size) / m_out_size;
		bbox_h[i] = (bounding_box_height[i] * m_heat_size) / m_out_size;
	}

	// 1. calculate mean and variance for each joint point
	convert_PCA_wld_to_BB();

	double k = pow(double(m_out_size) / double(m_heat_size), 2);
	int joint_i = 0;
	for (joint_i = 0; joint_i < JOINT_NUM; ++joint_i)
	{
		Eigen::Vector4f mean_18;
		if (!estimate_gauss_mean_covariance(joint_i, mean_18, joints_covariance_bb[joint_i]))
		{
			break;
		}

		xyz_18_96(mean_18, joints_means_bb[joint_i]);
		
		joints_covariance_bb[joint_i] *= k;	// covariance_18 in 18 x 18 3d space ---> covariance_96 in 96 x 96 3d space
		joints_inv_covariance_bb[joint_i] = joints_covariance_bb[joint_i].inverse();
	}
	if (joint_i<JOINT_NUM)
	{
		++estimate_gauss_failed_cnt;
		fuse_sub(estimate_xyz);
		return;
	}
	// 2. A*alpha = b
	cv::Mat A_mat(PCA_SZ, PCA_SZ, CV_32FC1);
	for (int i = 0; i < PCA_SZ; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			float a_ij = 0.0;
			for (int k = 0; k < JOINT_NUM; ++k)
			{
				Eigen::Vector3f tmp;
				tmp = joints_inv_covariance_bb[k] * pca_eigen_vecs_bb[i][k].head<3>();
				a_ij += pca_eigen_vecs_bb[j][k].head<3>().dot(tmp);
			}
			A_mat.at<float>(i, j) = a_ij;
			A_mat.at<float>(j, i) = a_ij;
		}
	}
	cv::Mat b_mat(PCA_SZ, 1, CV_32FC1);
	for (int i = 0; i < PCA_SZ; ++i)
	{
		float b_i = 0.0;
		for (int k = 0; k < JOINT_NUM; ++k)
		{
			Eigen::Vector3f tmp;
			tmp = joints_inv_covariance_bb[k] * pca_eigen_vecs_bb[i][k].head<3>();
			Eigen::Vector3f tmp_diff;
			for (int ki = 0; ki < 3; ++ki)
			{
				tmp_diff[ki] = joints_means_bb[k][ki] - pca_means_bb[k][ki];
			}
			b_i += tmp_diff.dot(tmp);
		}
		b_mat.at<float>(i, 0) = b_i;
	}
	cv::Mat alpha_mat = A_mat.inv()*b_mat;

	// 3. recover from PCA
	Eigen::Vector3f offset_vec(0.0, 0.0, 0.0);//(1.6, -1.2, -1.2);	//
	for (int i_joint = 0; i_joint < JOINT_NUM; ++i_joint)
	{
		Eigen::Vector4f estimate_pt(pca_means_bb[i_joint]);
		for (int i_pca = 0; i_pca < PCA_SZ; ++i_pca)
		{
			for (int i_xyz = 0; i_xyz < 3; ++i_xyz)
			{
				estimate_pt[i_xyz] += alpha_mat.at<float>(i_pca, 0)*pca_eigen_vecs_bb[i_pca][i_joint][i_xyz];
			}
		}
		//*
		//////////////draw estimate points on heat-maps//////////////
		Eigen::Vector4f estimate_pt_18;
		xyz_96_18(estimate_pt, estimate_pt_18);
		int u, v;
		_3d_xy(estimate_pt_18[0], estimate_pt_18[1], u, v);
		if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size)
		{
			std::cout << "Joint " << i_joint << ": xy out" << std::endl;
		}
		cv::circle(heatmaps_vec[i_joint * 3], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		_3d_yz(estimate_pt_18[1], estimate_pt_18[2], u, v);
		if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size)
		{
			std::cout << "Joint " << i_joint << ": yz out" << std::endl;
		}
		cv::circle(heatmaps_vec[i_joint * 3 + 1], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		_3d_zx(estimate_pt_18[2], estimate_pt_18[0], u, v);
		if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size)
		{
			std::cout << "Joint " << i_joint << ": zx out" << std::endl;
		}
		cv::circle(heatmaps_vec[i_joint * 3 + 2], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
		//////////////draw estimate points on heat-maps//////////////
		//*/
		Eigen::Vector4f tmp(estimate_pt);
		estimate_pt = depth_projector->get_relative_xform() * tmp;	// convert from BB cs to world cs
		for (int j = 0; j < 3; ++j)
		{
			estimate_xyz[3 * i_joint + j] = estimate_pt[j] - offset_vec[j];
		}
	}
}

void HeatmapFuser::fuse_sub(float* estimate_xyz)	// respectively optimization
{
	for (int i = 0; i < 3; ++i)
	{
		bbox_x[i] = (bounding_box_x[i] * m_heat_size) / m_out_size;
		bbox_y[i] = (bounding_box_y[i] * m_heat_size) / m_out_size;
		bbox_w[i] = (bounding_box_width[i] * m_heat_size) / m_out_size;
		bbox_h[i] = (bounding_box_height[i] * m_heat_size) / m_out_size;
	}
	Eigen::Vector3f offset_vec(0, 0, 0); //(-3.26, 4.24, -0.32);	//(-0.46716, -0.27824, -1.5545);	//(1.6, -1.2, -1.2);	//
	for (int i = 0; i < JOINT_NUM; ++i)
	{
		Eigen::Vector4f estimate_pt = estimate_joint_xyz(i);

		Eigen::Vector4f tmp(estimate_pt);
		estimate_pt = depth_projector->get_relative_xform() * tmp;
		for (int j = 0; j < 3; ++j)
		{
			estimate_xyz[3 * i + j] = estimate_pt[j] - offset_vec[j];
		}
	}
}

Eigen::Vector4f HeatmapFuser::estimate_joint_xyz(int joint_i)
{
	double h = 10;	// 50
	double lamda[3] = {3, 1, 1};

	double h_2 = pow(h, 2);
	double threshold = 1e-5;

	Eigen::Vector4f cur_pt(0.0,0.0,0.0,1.0);	// = get_start_point(joint_i);	// 
	
	Eigen::Vector4f pre_pt, diff;

	cv::Mat xy_8bit(m_heat_size, m_heat_size, CV_8UC1), heat_binary[3];
	cv::Mat yz_8bit(m_heat_size, m_heat_size, CV_8UC1);
	cv::Mat zx_8bit(m_heat_size, m_heat_size, CV_8UC1);
	for (int u = 0; u < m_heat_size; ++u)
	{
		for (int v = 0; v < m_heat_size; ++v)
		{
			xy_8bit.at<unsigned char>(v, u) = int(heatmaps_vec[joint_i * 3].at<float>(v, u) * 256);
			yz_8bit.at<unsigned char>(v, u) = int(heatmaps_vec[joint_i * 3 + 1].at<float>(v, u) * 256);
			zx_8bit.at<unsigned char>(v, u) = int(heatmaps_vec[joint_i * 3 + 2].at<float>(v, u) * 256);
		}
	}
	cv::threshold(xy_8bit, heat_binary[0], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(yz_8bit, heat_binary[1], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(zx_8bit, heat_binary[2], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	std::vector<cv::Point3d> pts_weights[3];		// pt.x save u, pt.y save v, pt.z save weight
	for (int i = 0; i < 3; ++i)
	{
		for (int u = 0; u < m_heat_size; ++u)
		{
			for (int v = 0; v < m_heat_size; ++v)
			{
				if (heat_binary[i].at<bool>(v, u))
				{
					cv::Point3d pt;
					pt.z = heatmaps_vec[joint_i * 3 + i].at<float>(v, u);
					_2d_3d(i, u, v, pt);
					pts_weights[i].push_back(pt);
				}
			}
		}
	}

	do
	{
		pre_pt = cur_pt;
		double x_num1 = 0.0, x_den1 = 0.0, y_num1 = 0.0, y_den1 = 0.0, z_num2 = 0.0, z_den2 = 0.0;
		double x_num3 = 0.0, x_den3 = 0.0, y_num2 = 0.0, y_den2 = 0.0, z_num3 = 0.0, z_den3 = 0.0;

		for (int j = 0; j < pts_weights[0].size(); ++j)		// xy
		{
			double exp_term = 
				exp(-(pow(cur_pt[0] - pts_weights[0][j].x, 2) + pow(cur_pt[1] - pts_weights[0][j].y, 2)) / h_2);
			double tmp = pts_weights[0][j].z * exp_term;
			x_den1 += tmp;
			y_den1 += tmp;
			x_num1 += tmp*pts_weights[0][j].x;
			y_num1 += tmp*pts_weights[0][j].y;
		}
		for (int j = 0; j < pts_weights[1].size(); ++j)		// yz
		{
			double exp_term =
				exp(-(pow(cur_pt[1] - pts_weights[1][j].x, 2) + pow(cur_pt[2] - pts_weights[1][j].y, 2)) / h_2);
			double tmp = pts_weights[1][j].z * exp_term;
			y_den2 += tmp;
			z_den2 += tmp;
			y_num2 += tmp*pts_weights[1][j].x;
			z_num2 += tmp*pts_weights[1][j].y;
		}
		for (int j = 0; j < pts_weights[2].size(); ++j)		// zx
		{
			double exp_term =
				exp(-(pow(cur_pt[2] - pts_weights[2][j].x, 2) + pow(cur_pt[0] - pts_weights[2][j].y, 2)) / h_2);
			double tmp = pts_weights[2][j].z * exp_term;
			z_den3 += tmp;
			x_den3 += tmp;
			z_num3 += tmp*pts_weights[2][j].x;
			x_num3 += tmp*pts_weights[2][j].y;
		}

		cur_pt[0] = (lamda[0] * x_num1 + lamda[2] * x_num3) / (lamda[0] * x_den1 + lamda[2] * x_den3);
		cur_pt[1] = (lamda[0] * y_num1 + lamda[1] * y_num2) / (lamda[0] * y_den1 + lamda[1] * y_den2);
		cur_pt[2] = (lamda[1] * z_num2 + lamda[2] * z_num3) / (lamda[1] * z_den2 + lamda[2] * z_den3);
		//cur_pt[2] = (z_num2 + z_num3) / (z_den2 + z_den3);

		diff = pre_pt - cur_pt;
	} while (diff.squaredNorm()>threshold);
	//*/
	int u, v;
	_3d_xy(cur_pt[0], cur_pt[1], u, v);
	if (u<0 || v<0 || u>=m_heat_size || v>=m_heat_size)
	{
		std::cout << "Joint " << joint_i << ": xy out" << std::endl;
	}
	cv::circle(heatmaps_vec[joint_i * 3], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
	_3d_yz(cur_pt[1], cur_pt[2], u, v);
	if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size)
	{
		std::cout << "Joint " << joint_i << ": yz out" << std::endl;
	}
	cv::circle(heatmaps_vec[joint_i * 3 + 1], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);
	_3d_zx(cur_pt[2], cur_pt[0], u, v);
	if (u < 0 || v < 0 || u >= m_heat_size || v >= m_heat_size)
	{
		std::cout << "Joint " << joint_i << ": zx out" << std::endl;
	}
	cv::circle(heatmaps_vec[joint_i * 3 + 2], cv::Point(u, v), 2, cv::Scalar(0, 0, 255, 0), CV_FILLED, CV_AA, 0);

	Eigen::Vector4f pt_96;
	xyz_18_96(cur_pt, pt_96);

	return pt_96;
}

bool HeatmapFuser::estimate_gauss_mean_covariance(int joint_i, Eigen::Vector4f& mean_18, Eigen::Matrix3f& covariance_18)
{
	cv::Mat xy_8bit(m_heat_size, m_heat_size, CV_8UC1), heat_binary[3];
	cv::Mat yz_8bit(m_heat_size, m_heat_size, CV_8UC1);
	cv::Mat zx_8bit(m_heat_size, m_heat_size, CV_8UC1);
	for (int u = 0; u < m_heat_size; ++u)
	{
		for (int v = 0; v < m_heat_size; ++v)
		{
			xy_8bit.at<unsigned char>(v, u) = int(heatmaps_vec[joint_i * 3].at<float>(v, u) * 256);
			yz_8bit.at<unsigned char>(v, u) = int(heatmaps_vec[joint_i * 3 + 1].at<float>(v, u) * 256);
			zx_8bit.at<unsigned char>(v, u) = int(heatmaps_vec[joint_i * 3 + 2].at<float>(v, u) * 256);
		}
	}
	cv::threshold(xy_8bit, heat_binary[0], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(yz_8bit, heat_binary[1], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(zx_8bit, heat_binary[2], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	// 1. get sample points
	std::vector<Eigen::Vector4f> xyz_weights;		// xyz_weights[0]->x, xyz_weights[1]->y, xyz_weights[2]->z, xyz_weights[3]->weight
	double z_value[m_heat_size];
	for (int i_z = 0; i_z < m_heat_size; ++i_z)
	{
		z_value[i_z] = depth_projector->get_z_length() * (i_z + 0.5) / double(m_out_size);
	}
	for (int u = 0; u < m_heat_size; ++u)
	{
		for (int v = 0; v < m_heat_size; ++v)
		{
			if (!heat_binary[0].at<bool>(v, u))
			{
				continue;
			}
			cv::Point3d pt;
			pt.z = heatmaps_vec[joint_i * 3].at<float>(v, u);
			xy_3d(u, v, pt.x, pt.y);
			//_2d_3d(0, u, v, pt);
			for (int i_z = 0; i_z < m_heat_size; ++i_z)
			{
				int u_yz, v_yz;
				_3d_yz(pt.y, z_value[i_z], u_yz, v_yz);
				if (u_yz < 0 || u_yz >= m_heat_size || v_yz < 0 || v_yz >= m_heat_size || !heat_binary[1].at<bool>(v_yz, u_yz))
				{
					continue;
				}
				float conf_yz = heatmaps_vec[joint_i * 3 + 1].at<float>(v_yz, u_yz);

				int u_zx, v_zx;
				_3d_zx(z_value[i_z], pt.x, u_zx, v_zx);
				if (u_zx < 0 || u_zx >= m_heat_size || v_zx < 0 || v_zx >= m_heat_size || !heat_binary[2].at<bool>(v_zx, u_zx))
				{
					continue;
				}
				float conf_zx = heatmaps_vec[joint_i * 3 + 2].at<float>(v_zx, u_zx);

				xyz_weights.push_back(Eigen::Vector4f(pt.x, pt.y, z_value[i_z], fuse_confidence(pt.z, conf_yz, conf_zx)));
			}
		}
	}

	if (xyz_weights.size() == 0)
	{
		return false;
	}

	// 2. get mean
	Eigen::Vector4f xyz_sum(0.0, 0.0, 0.0, 1.0);
	float conf_sum = 0.0;

	for (int xyz_i = 0; xyz_i < xyz_weights.size(); ++xyz_i)
	{
		conf_sum += xyz_weights[xyz_i][3];
		for (int i = 0; i < 3; ++i)
		{
			xyz_sum[i] += xyz_weights[xyz_i][3] * xyz_weights[xyz_i][i];
		}
	}
	for (int i = 0; i < 3; ++i)
	{
		mean_18[i] = xyz_sum[i] / conf_sum;
	}
	mean_18[3] = 1.0;

	// 3. get covariance
	covariance_18 = Eigen::Matrix3f::Zero();
	Eigen::Vector4f xyz_var(0.0, 0.0, 0.0, 1.0);
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			for (int xyz_i = 0; xyz_i < xyz_weights.size(); ++xyz_i)
			{
				covariance_18(i, j) += xyz_weights[xyz_i][3] * (xyz_weights[xyz_i][i] - mean_18[i])
															 * (xyz_weights[xyz_i][j] - mean_18[j]);
			}
			covariance_18(i, j) /= conf_sum;
			covariance_18(j, i) = covariance_18(i, j);
		}
	}

	float var_low_limit = 1e-3;
	for (int i = 0; i < 3; ++i)
	{
		if (covariance_18(i,i) < var_low_limit)
		{
			covariance_18(i, i) = var_low_limit;
		}
	}
	if (covariance_18.determinant() <= 1)
	{
		return false;
	}
	return true;
}

void HeatmapFuser::_2d_3d(int view_type, int u, int v, cv::Point3d& pt)
{
	if (view_type==0)
	{
		xy_3d(u, v, pt.x, pt.y);
	}
	else if (view_type == 1)
	{
		yz_3d(u, v, pt.x, pt.y);
	}
	else if (view_type == 2)
	{
		zx_3d(u, v, pt.x, pt.y);
	}
}

void HeatmapFuser::xy_3d(int u, int v, double& x, double& y)
{
	x = (u - bbox_x[0] + 1) / proj_k[0];
	y = (v - bbox_y[0] + 1) / proj_k[0];
}

void HeatmapFuser::yz_3d(int u, int v, double& y, double& z)
{
	y = (u - bbox_x[1] + 1) / proj_k[1];
	z = (v - bbox_y[1] + 1) / proj_k[1];
}

void HeatmapFuser::zx_3d(int u, int v, double& z, double& x)
{
	z = (u - bbox_x[2] + 1) / proj_k[2];
	x = (v - bbox_y[2] + 1) / proj_k[2];
}

void HeatmapFuser::xyz_18_96(const Eigen::Vector4f& xyz_18, Eigen::Vector4f& xyz_96)
{
	xyz_96[0] = (m_out_size*xyz_18[0]) / m_heat_size - depth_projector->get_x_length()/ 2.0;
	xyz_96[1] = -(m_out_size*xyz_18[1]) / m_heat_size + depth_projector->get_y_length() / 2.0;
	xyz_96[2] = (m_out_size*xyz_18[2]) / m_heat_size - depth_projector->get_z_length() / 2.0;
	xyz_96[3] = 1.0;
}

void HeatmapFuser::xyz_96_18(const Eigen::Vector4f& xyz_96, Eigen::Vector4f& xyz_18)
{
	xyz_18[0] = (m_heat_size * (xyz_96[0] + depth_projector->get_x_length() / 2.0)) / m_out_size;
	xyz_18[1] = -(m_heat_size * (xyz_96[1] - depth_projector->get_y_length() / 2.0)) / m_out_size;
	xyz_18[2] = (m_heat_size * (xyz_96[2] + depth_projector->get_z_length() / 2.0)) / m_out_size;
	xyz_18[3] = 1.0;
}

void HeatmapFuser::_3d_xy(double x, double y, int& u, int& v)
{
	u = int(proj_k[0] * x + bbox_x[0]);
	v = int(proj_k[0] * y + bbox_y[0]);

	//u = int_rounding(proj_k[0] * x + bbox_x[0]);
	//v = int_rounding(proj_k[0] * y + bbox_y[0]);
}

void HeatmapFuser::_3d_yz(double y, double z, int& u, int& v)
{
	u = int(proj_k[1] * y + bbox_x[1]);
	v = int(proj_k[1] * z + bbox_y[1]);

	//u = int_rounding(proj_k[1] * y + bbox_x[1]);
	//v = int_rounding(proj_k[1] * z + bbox_y[1]);
}

void HeatmapFuser::_3d_zx(double z, double x, int& u, int& v)
{
	u = int(proj_k[2] * z + bbox_x[2]);
	v = int(proj_k[2] * x + bbox_y[2]);

	//u = int_rounding(proj_k[2] * z + bbox_x[2]);
	//v = int_rounding(proj_k[2] * x + bbox_y[2]);
}

void HeatmapFuser::convert_PCA_wld_to_BB()
{
	Eigen::Matrix4f trans_bb_wld;
	trans_bb_wld = depth_projector->get_relative_xform().inverse();

	for (int i = 0; i < 3; ++i)
	{
		trans_bb_wld(i, 3) = 0.0;
	}

	for (int i_pca = 0; i_pca < PCA_SZ; ++i_pca)
	{
		for (int i_joint = 0; i_joint < JOINT_NUM; ++i_joint)
		{
			Eigen::Vector4f eigen_wld(0.0, 0.0, 0.0, 1.0);
			for (int i = 0; i < 3; ++i)
			{
				eigen_wld[i] = fuser_pca.eigenvectors.at<float>(i_pca, i_joint * 3 + i);
			}
			pca_eigen_vecs_bb[i_pca][i_joint] = trans_bb_wld * eigen_wld;
		}
	}

	for (int i_joint = 0; i_joint < JOINT_NUM; ++i_joint)
	{
		Eigen::Vector4f mean_wld(0.0, 0.0, 0.0, 1.0);
		for (int i = 0; i < 3; ++i)
		{
			mean_wld[i] = fuser_pca.mean.at<float>(0, i_joint * 3 + i);
		}
		pca_means_bb[i_joint] = trans_bb_wld * mean_wld;
	}
}

float HeatmapFuser::fuse_confidence(float conf_xy, float conf_yz, float conf_zx)
{
	//return pow(conf_xy, 1 / lamda[0]) * pow(conf_yz, 1 / lamda[1]) * pow(conf_zx, 1 / lamda[2]);
	return conf_xy * conf_yz * conf_zx;
}