#include "depth_projector.h"

DepthProjector::DepthProjector(int x_res, int y_res, double focal_length, int out_size, int heat_size)
{   
	m_x_res = x_res;
	m_y_res = y_res;
	m_focal_length = focal_length;
	m_out_size = out_size;
	m_heat_size = heat_size;

    m_x_length = 0.0f;
    m_y_length = 0.0f;
    m_z_length = 0.0f;
}

bool DepthProjector::load_depth_image(const cv::Mat &depth_image, const cv::Rect &bbox)
{
	int num_elements = bbox.width * bbox.height;
	std::vector<Eigen::Vector3f> pcd(num_elements);

	depth_image.forEach<float>([&](float &depth, const int position[]) -> void {
        int x = position[1], y = position[0];

		Eigen::Vector3f uv_depth(x + bbox.x, y + bbox.y, depth);
		Eigen::Vector3f world_co(-(m_x_res / 2.0 - uv_depth.x()) * uv_depth.z() / m_focal_length,
								 -(uv_depth.y() - m_y_res / 2.0) * uv_depth.z() / m_focal_length,
								 uv_depth.z());

		pcd[y * bbox.width + x] = world_co;
    });

	// Calculate center point
	m_xyz_data.clear();
	m_center_pt = Eigen::Vector3f(0.0, 0.0, 0.0);
	for (int i = 0; i < num_elements; i++) {
		Eigen::Vector3f &point = pcd[i];
		
		// Skip if the depth is 0
		if (fabs(point.z()) < FLT_EPSILON) {
			continue;
		}

		Eigen::Vector4f data_pt(point.x(), point.y(), point.z(), 1.0f);
		m_xyz_data.push_back(data_pt);
		for (int j = 0; j < 3; j++)
		{
			m_center_pt[j] += data_pt[j];	
		}		
	}

	if (m_xyz_data.size() > 0) {
		m_center_pt /= m_xyz_data.size();
		return true;
	} else {
		return false;
	}
}

void DepthProjector::load_ground_truth(const std::array<Eigen::Vector3f, 21> &gt)
{
	m_gt_xyz_data.clear();

	for (int i = 0; i < 21; i++) {
		Eigen::Vector3f joint = gt[i];
		Eigen::Vector4f gt_data(joint.x(), joint.y(), joint.z(), 1.0f);
		m_gt_xyz_data.push_back(gt_data);	
	}
}

bool DepthProjector::load_data(const cv::Mat &depth_image, const cv::Rect &bbox, const std::array<Eigen::Vector3f, 21> &gt)
{
	if (!load_depth_image(depth_image, bbox)) {
		return false;
	}

	load_ground_truth(gt);

	create_obb();

	create_projections();
	create_heatmap_uvs();
	return true;
}

void DepthProjector::create_obb()
{
	// 1. calculate covariance matrix
	Eigen::Matrix3f cov;
	for (int j = 0; j < 3; j++)
	{
		for (int k = j; k < 3; k++)
		{
			cov(j,k) = 0.0;

			for (int i = 0; i < m_xyz_data.size(); i++)
			{
				cov(j, k) += m_xyz_data[i](j) * m_xyz_data[i](k);		//c[j] * c[k];
			}

			cov(j, k) /= m_xyz_data.size();	
			cov(j, k) -= (m_center_pt[j] * m_center_pt[k]);
		}
	}

	for (int j = 0; j < 3; j++)
	{
		for (int k = 0; k < j; k++)
		{
			cov(j, k) = cov(k, j);
		}
	}

	// 2. calculate Eigen vectors to determine x_axis, y_axis, z_axis
	Eigen::EigenSolver<Eigen::Matrix3f> eigensolver(cov);
	auto eigenvectors = eigensolver.eigenvectors().real();
	Eigen::Vector3f x_axis = eigenvectors.col(0);
	Eigen::Vector3f y_axis = eigenvectors.col(1);
	Eigen::Vector3f z_axis = eigenvectors.col(2);
	if (x_axis.norm() < FLT_EPSILON || y_axis.norm() < FLT_EPSILON || z_axis.norm() < FLT_EPSILON) {
		x_axis = Eigen::Vector3f(1.0, 0.0, 0.0);
		y_axis = Eigen::Vector3f(0.0, 1.0, 0.0);
		z_axis = Eigen::Vector3f(0.0, 0.0, 1.0);
	} else {
		x_axis.normalize();
		y_axis.normalize();
		z_axis = y_axis.cross(x_axis);
		if (z_axis[2] < 0)
		{
			z_axis *= -1;
			if (x_axis[1] < 0) {
				x_axis = z_axis.cross(y_axis);
			} else {
				y_axis = x_axis.cross(z_axis);
			}
		}
	}

	// 3. calculate length, width, height
	std::vector<double> a, b, c;
	for (int i = 0; i < m_xyz_data.size(); i++)
	{
		Eigen::Vector3f ov = m_xyz_data[i].head<3>() - m_center_pt;
		a.push_back(ov.dot(x_axis));
		b.push_back(ov.dot(y_axis));
		c.push_back(ov.dot(z_axis));
	}

	double a_min = *std::min_element(a.begin(), a.end());
	double a_max = *std::max_element(a.begin(), a.end());
	double b_min = *std::min_element(b.begin(), b.end());
	double b_max = *std::max_element(b.begin(), b.end());
	double c_min = *std::min_element(c.begin(), c.end());
	double c_max = *std::max_element(c.begin(), c.end());

	double scale = 1.02;	// scale coefficient
	m_x_length = (a_max - a_min) * scale;
	m_y_length = (b_max - b_min) * scale;
	m_z_length = (c_max - c_min) * scale;

	// 4. adjust center point
	m_center_pt += x_axis * (a_min + a_max) / 2.0;
	m_center_pt += y_axis * (b_min + b_max) / 2.0;
	m_center_pt += z_axis * (c_min + c_max) / 2.0;

	// 5. get m_relative_xform
	m_relative_xform << x_axis[0], y_axis[0], z_axis[2], m_center_pt[0],
					  x_axis[1], y_axis[1], z_axis[1], m_center_pt[1],
					  x_axis[2], y_axis[2], z_axis[2], m_center_pt[2],
					  0.0, 0.0, 0.0, 1.0;

	// 6. transform 3d points from world coords to bbox coords
	Eigen::Matrix4f inv_relative_xform = m_relative_xform.inverse();
	for (int i = 0; i < m_xyz_data.size(); i++) {
		m_xyz_data[i] = inv_relative_xform * m_xyz_data[i];
	}

	for (int i = 0; i < m_gt_xyz_data.size(); i++) {
		m_gt_xyz_data[i] = inv_relative_xform * m_gt_xyz_data[i];
	}
}

void DepthProjector::create_projections()
{
	int pad = int(m_out_size * 0.1);

	// 0. x-y
	m_projections[0] = cv::Mat::ones(m_out_size, m_out_size, CV_32F);

	if (m_x_length >= m_y_length) {
		m_projected_bbox[0].width = m_out_size - (pad << 1);
		m_projected_bbox[0].height = std::max(1.0, std::round((m_projected_bbox[0].width*m_y_length) / m_x_length));
		m_proj_k[0] = float(m_projected_bbox[0].width) / m_x_length;
	} else {
		m_projected_bbox[0].height = m_out_size - (pad << 1);
		m_projected_bbox[0].width = std::max(1.0, std::round((m_projected_bbox[0].height*m_x_length) / m_y_length));
		m_proj_k[0] = float(m_projected_bbox[0].height) / m_y_length;
	}

	m_projected_bbox[0].x = ((m_out_size - m_projected_bbox[0].width) >> 1);
	m_projected_bbox[0].y = ((m_out_size - m_projected_bbox[0].height) >> 1);
	cv::Mat xy_roi(m_projections[0], m_projected_bbox[0]);

	for (int i = 0; i < m_xyz_data.size(); i++) {
		int xy_u = std::clamp((int) std::round(m_proj_k[0]*(m_xyz_data[i][0] + m_x_length / 2.0)), 0, m_projected_bbox[0].width - 1);
		int xy_v = std::clamp((int) std::round(m_proj_k[0]*(-m_xyz_data[i][1] + m_y_length / 2.0)), 0, m_projected_bbox[0].height - 1);

		// normalize 0-1 and set the nearest point
		float norm_depth = std::max(0.0, (m_xyz_data[i][2] + m_z_length / 2.0) / m_z_length);
		if (xy_roi.at<float>(xy_v, xy_u) > norm_depth) {
			xy_roi.at<float>(xy_v, xy_u) = norm_depth;
		}
	}

	cv::medianBlur(m_projections[0], m_projections[0], 5);

	// 1. y-z
	m_projections[1] = cv::Mat::ones(m_out_size, m_out_size, CV_32F);

	if (m_y_length >= m_z_length) {
		m_projected_bbox[1].width = m_out_size - (pad << 1);
		m_projected_bbox[1].height = std::max(1.0, std::round((m_projected_bbox[1].width*m_z_length) / m_y_length));
		m_proj_k[1] = float(m_projected_bbox[1].width) / m_y_length;
	} else {
		m_projected_bbox[1].height = m_out_size - (pad << 1);
		m_projected_bbox[1].width = std::max(1.0, std::round((m_projected_bbox[1].height*m_y_length) / m_z_length));
		m_proj_k[1] = float(m_projected_bbox[1].height) / m_z_length;
	}

	m_projected_bbox[1].x = ((m_out_size - m_projected_bbox[1].width) >> 1);
	m_projected_bbox[1].y = ((m_out_size - m_projected_bbox[1].height) >> 1);
	cv::Mat yz_roi(m_projections[1], m_projected_bbox[1]);

	for (int i = 0; i < m_xyz_data.size(); ++i) {
		int yz_u = std::clamp((int) std::round(m_proj_k[1]*(-m_xyz_data[i][1] + m_y_length / 2.0)), 0, m_projected_bbox[1].width - 1);
		int yz_v = std::clamp((int) std::round(m_proj_k[1]*(m_xyz_data[i][2] + m_z_length / 2.0)), 0, m_projected_bbox[1].height - 1);

		// normalize 0-1 and set the nearest point
		float norm_depth = std::max(0.0, (-m_xyz_data[i][0] + m_x_length / 2.0) / m_x_length);
		if (yz_roi.at<float>(yz_v, yz_u) > norm_depth) {
			yz_roi.at<float>(yz_v, yz_u) = norm_depth;
		}
	}

	cv::medianBlur(m_projections[1], m_projections[1], 5);

	// 2. z-x
	m_projections[2] = cv::Mat::ones(m_out_size, m_out_size, CV_32F);

	if (m_z_length >= m_x_length) {
		m_projected_bbox[2].width = m_out_size - (pad << 1);
		m_projected_bbox[2].height = std::max(1.0, std::round((m_projected_bbox[2].width*m_x_length) / m_z_length));
		m_proj_k[2] = float(m_projected_bbox[2].width) / m_z_length;
	} else {
		m_projected_bbox[2].height = m_out_size - (pad << 1);
		m_projected_bbox[2].width = std::max(1.0, std::round((m_projected_bbox[2].height*m_z_length) / m_x_length));
		m_proj_k[2] = float(m_projected_bbox[2].height) / m_x_length;
	}

	m_projected_bbox[2].x = ((m_out_size - m_projected_bbox[2].width) >> 1);
	m_projected_bbox[2].y = ((m_out_size - m_projected_bbox[2].height) >> 1);
	cv::Mat zx_roi(m_projections[2], m_projected_bbox[2]);

	for (int i = 0; i < m_xyz_data.size(); ++i) {
		int zx_u = std::clamp((int) std::round(m_proj_k[2]*(m_xyz_data[i][2] + m_z_length / 2.0)), 0, m_projected_bbox[2].width - 1);
		int zx_v = std::clamp((int) std::round(m_proj_k[2]*(m_xyz_data[i][0] + m_x_length / 2.0)), 0, m_projected_bbox[2].height - 1);

		// normalize 0-1 and set the nearest point
		float norm_depth = std::max(0.0, (-m_xyz_data[i][1] + m_y_length / 2.0) / m_y_length);
		if (zx_roi.at<float>(zx_v, zx_u) > norm_depth) {
			zx_roi.at<float>(zx_v, zx_u) = norm_depth;
		}
	}
	
	cv::medianBlur(m_projections[2], m_projections[2], 5);
}

void DepthProjector::create_heatmap_uvs()
{
	for (int i = 0; i < 21; i++) {
		// 0. x-y
		m_joint_uvs[0][i].x = std::clamp(std::round(m_proj_k[0]*(m_gt_xyz_data[i][0] + m_x_length / 2.0)) + m_projected_bbox[0].x,
										 0.0, m_out_size - 1.0);
		m_joint_uvs[0][i].y = std::clamp(std::round(m_proj_k[0]*(-m_gt_xyz_data[i][1] + m_y_length / 2.0)) + m_projected_bbox[0].y,
										 0.0, m_out_size - 1.0);

		// 1. y-z
		m_joint_uvs[1][i].x = std::clamp(std::round(m_proj_k[1]*(-m_gt_xyz_data[i][1] + m_y_length / 2.0)) + m_projected_bbox[1].x,
										 0.0, m_out_size - 1.0);
		m_joint_uvs[1][i].y = std::clamp(std::round(m_proj_k[1]*(m_gt_xyz_data[i][2] + m_z_length / 2.0)) + m_projected_bbox[1].y,
										 0.0, m_out_size - 1.0);

		// 2. z-x
		m_joint_uvs[2][i].x = std::clamp(std::round(m_proj_k[2]*(m_gt_xyz_data[i][2] + m_z_length / 2.0)) + m_projected_bbox[2].x,
										 0.0, m_out_size - 1.0);
		m_joint_uvs[2][i].y = std::clamp(std::round(m_proj_k[2]*(m_gt_xyz_data[i][0] + m_x_length / 2.0)) + m_projected_bbox[2].y,
										 0.0, m_out_size - 1.0);
	}
}
