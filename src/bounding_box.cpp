#include "bounding_box.h"

BoundingBox::BoundingBox()
{   
    x_length = 0.0f;
    y_length = 0.0f;
    z_length = 0.0f;
}

bool BoundingBox::create_obb(float* xyz_data, int data_num, float* GT_xyz_data, int GT_data_num)
{
	m_xyz_data.clear();
	m_gt_xyz_data.clear();

	// 1. calculate center point
	Eigen::Vector3f center_pt(0.0, 0.0, 0.0);
	for (int i = 0; i < data_num; i += 3)
	{
		if (fabs(xyz_data[i + 2]) < FLT_EPSILON)	// depth == 0.0 then skip
		{
			continue;
		}

		Eigen::Vector4f data_pt(xyz_data[i], xyz_data[i + 1], xyz_data[i + 2], 1.0f);
		m_xyz_data.push_back(data_pt);

		for (int j = 0; j < 3; j++)
		{
			center_pt[j] += data_pt[j];	
		}		
	}

	for (int i = 0; i < GT_data_num; i += 3)
	{
		Eigen::Vector4f gt_data_pt(GT_xyz_data[i], GT_xyz_data[i + 1], GT_xyz_data[i + 2], 1.0f);
		m_gt_xyz_data.push_back(gt_data_pt);
	}

	int pt_num = m_xyz_data.size();
	if (pt_num <= 0)
	{
		return false;
	}

	for (int i = 0; i < 3; i++)
	{
		center_pt[i] /= pt_num;
	}

	// 2. calculate covariance matrix
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

			cov(j, k) /= pt_num;	
			cov(j, k) -= (center_pt[j] * center_pt[k]);
		}
	}

	for (int j = 0; j < 3; j++)
	{
		for (int k = 0; k < j; k++)
		{
			cov(j, k) = cov(k, j);
		}
	}

	// 3. calculate Eigen vectors to determine x_axis, y_axis, z_axis
	Eigen::Vector3f x_axis, y_axis, z_axis;
	Eigen::Matrix3f eigenvector_matrix;	// Eigen vectors
	double eigenvalue_array[3];		// Eigen values

	jacobbi(cov, eigenvector_matrix, eigenvalue_array);
	x_axis[0] = eigenvector_matrix(0,0);
	x_axis[1] = eigenvector_matrix(1,0);
	x_axis[2] = eigenvector_matrix(2,0);
	y_axis[0] = eigenvector_matrix(0,1);
	y_axis[1] = eigenvector_matrix(1,1);
	y_axis[2] = eigenvector_matrix(2,1);
	z_axis[0] = eigenvector_matrix(0,2);
	z_axis[1] = eigenvector_matrix(1,2);
	z_axis[2] = eigenvector_matrix(2,2);
	if (x_axis.norm() < FLT_EPSILON || y_axis.norm() < FLT_EPSILON || z_axis.norm() < FLT_EPSILON)
	{
		x_axis = Eigen::Vector3f(1.0, 0.0, 0.0);
		y_axis = Eigen::Vector3f(0.0, 1.0, 0.0);
		z_axis = Eigen::Vector3f(0.0, 0.0, 1.0);
	}
	else
	{
		x_axis.normalize();
		y_axis.normalize();
		z_axis = y_axis.cross(x_axis);
		if (z_axis[2] <0)
		{
			z_axis *= -1;
			if (x_axis[1] < 0) {
				x_axis = z_axis.cross(y_axis);
			} else {
				y_axis = x_axis.cross(z_axis);
			}
		}
	}

	// 4. calculate length, width, height
	std::vector<double> a, b, c;
	for (int i = 0; i < pt_num; i++)
	{
		Eigen::Vector3f ov = m_xyz_data[i].head<3>() - center_pt;
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
	x_length = (a_max - a_min) * scale;
	y_length = (b_max - b_min) * scale;
	z_length = (c_max - c_min) * scale;

	// 5. adjust center point
	center_pt += x_axis * (a_min + a_max) / 2.0;
	center_pt += y_axis * (b_min + b_max) / 2.0;
	center_pt += z_axis * (c_min + c_max) / 2.0;

	// 6. get relative_xform
	relative_xform << x_axis[0], x_axis[1], x_axis[2], 0.0,
					  y_axis[0], y_axis[1], y_axis[2], 0.0,
					  z_axis[0], z_axis[1], z_axis[2], 0.0,
					  center_pt[0], center_pt[1], center_pt[2], 1.0;

	// 7. transform 3d points from world cs to BB cs
	Eigen::Matrix4f inv_relative_xform = relative_xform.inverse();
	for (int i = 0; i < m_xyz_data.size(); i++)
	{
		m_xyz_data[i] = inv_relative_xform * m_xyz_data[i];
	}

	for (int i = 0; i < m_gt_xyz_data.size(); i++)
	{
		m_gt_xyz_data[i] = inv_relative_xform * m_gt_xyz_data[i];
	}

	return true;
}

bool BoundingBox::project_direct(cv::Mat* proj_im, cv::Point2f proj_uv[][JOINT_NUM], int sz)
{
	for (int i = 0; i < 3; i++)
	{
		proj_im[i].release();
	}

	// 0. x-y
	int pad = int(sz*0.1);
	proj_im[0] = cv::Mat::ones(sz, sz, CV_32F);
	if (x_length >= y_length)
	{
		m_projected_bbox[0].width = sz - 2 * pad;
		m_projected_bbox[0].height = std::round((m_projected_bbox[0].width * (y_length) / x_length));
		m_proj_k[0] = float(m_projected_bbox[0].width) / x_length;
	} else {
		m_projected_bbox[0].height = sz - (pad << 1);
		m_projected_bbox[0].width = std::round((m_projected_bbox[0].height*x_length) / y_length);
		m_proj_k[0] = float(m_projected_bbox[0].height) / y_length;
	}
	m_projected_bbox[0].x = ((sz - m_projected_bbox[0].width) >> 1);
	m_projected_bbox[0].y = ((sz - m_projected_bbox[0].height) >> 1);
	cv::Mat xy_roi = proj_im[0](m_projected_bbox[0]);

	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		int xy_u = std::round(m_proj_k[0]*(m_xyz_data[i][0] + x_length / 2.0));
		int xy_v = std::round(m_proj_k[0]*(-m_xyz_data[i][1] + y_length / 2.0));

		if (xy_u < 0)	xy_u = 0;
		else if (xy_u >= m_projected_bbox[0].width)		xy_u = m_projected_bbox[0].width - 1;

		if (xy_v < 0)	xy_v = 0;
		else if (xy_v >= m_projected_bbox[0].height)	xy_v = m_projected_bbox[0].height - 1;

		float norm_depth = (m_xyz_data[i][2] + z_length / 2.0) / z_length;	// normalize 0-1
		if (norm_depth<0)
		{
			norm_depth = 0.0f;
		}
		if (xy_roi.at<float>(xy_v, xy_u) > norm_depth)		// set the nearest point
		{
			xy_roi.at<float>(xy_v, xy_u) = norm_depth;
		}
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		proj_uv[0][i].x = std::round(m_proj_k[0]*(m_gt_xyz_data[i][0] + x_length / 2.0)) + m_projected_bbox[0].x;
		proj_uv[0][i].y = std::round(m_proj_k[0]*(-m_gt_xyz_data[i][1] + y_length / 2.0)) + m_projected_bbox[0].y;

		if (proj_uv[0][i].x < 0)	proj_uv[0][i].x = 0;
		else if (proj_uv[0][i].x >= sz)	proj_uv[0][i].x = sz - 1;

		if (proj_uv[0][i].y < 0)	proj_uv[0][i].y = 0;
		else if (proj_uv[0][i].y >= sz)	proj_uv[0][i].y = sz - 1;
	}

	// 1. y-z
	//cv::Rect yz_bounding_box;
	//float yz_k = 0.0;
	proj_im[1] = cv::Mat::ones(sz, sz, CV_32F);
	if (y_length >= z_length)
	{
		m_projected_bbox[1].width = sz - (pad << 1);
		m_projected_bbox[1].height = std::round((m_projected_bbox[1].width*z_length) / y_length);
		m_proj_k[1] = float(m_projected_bbox[1].width) / y_length;
	}
	else
	{
		m_projected_bbox[1].height = sz - (pad << 1);
		m_projected_bbox[1].width = std::round((m_projected_bbox[1].height*y_length) / z_length);
		m_proj_k[1] = float(m_projected_bbox[1].height) / z_length;
	}
	m_projected_bbox[1].x = ((sz - m_projected_bbox[1].width) >> 1);
	m_projected_bbox[1].y = ((sz - m_projected_bbox[1].height) >> 1);
	cv::Mat yz_roi = proj_im[1](m_projected_bbox[1]);

	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		int yz_u = std::round(m_proj_k[1]*(-m_xyz_data[i][1] + y_length / 2.0));
		int yz_v = std::round(m_proj_k[1]*(m_xyz_data[i][2] + z_length / 2.0));
		
		if (yz_u < 0)	yz_u = 0;
		else if (yz_u >= m_projected_bbox[1].width)		yz_u = m_projected_bbox[1].width - 1;

		if (yz_v < 0)	yz_v = 0;
		else if (yz_v >= m_projected_bbox[1].height)	yz_v = m_projected_bbox[1].height - 1;

		float norm_depth = (-m_xyz_data[i][0] + x_length / 2.0) / x_length;//(m_xyz_data[i][0] + x_length / 2.0) / x_length;	// normalize 0-1
		if (norm_depth<0)
		{
			norm_depth = 0.0f;
		}
		if (yz_roi.at<float>(yz_v, yz_u) > norm_depth)		// set the nearest point
		{
			yz_roi.at<float>(yz_v, yz_u) = norm_depth;
		}
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		proj_uv[1][i].x = std::round(m_proj_k[1]*(-m_gt_xyz_data[i][1] + y_length / 2.0)) + m_projected_bbox[1].x;
		proj_uv[1][i].y = std::round(m_proj_k[1]*(m_gt_xyz_data[i][2] + z_length / 2.0)) + m_projected_bbox[1].y;

		if (proj_uv[1][i].x < 0)	proj_uv[1][i].x = 0;
		else if (proj_uv[1][i].x >= sz)	proj_uv[1][i].x = sz - 1;

		if (proj_uv[1][i].y < 0)	proj_uv[1][i].y = 0;
		else if (proj_uv[1][i].y >= sz)	proj_uv[1][i].y = sz - 1;
	}

	// 2. z-x
	//cv::Rect zx_bounding_box;
	//float zx_k = 0.0;
	proj_im[2] = cv::Mat::ones(sz, sz, CV_32F);
	if (z_length >= x_length)
	{
		m_projected_bbox[2].width = sz - (pad << 1);
		m_projected_bbox[2].height = std::round((m_projected_bbox[2].width*x_length) / z_length);
		m_proj_k[2] = float(m_projected_bbox[2].width) / z_length;
	}
	else
	{
		m_projected_bbox[2].height = sz - (pad << 1);
		m_projected_bbox[2].width = std::round((m_projected_bbox[2].height*z_length) / x_length);
		m_proj_k[2] = float(m_projected_bbox[2].height) / x_length;
	}
	m_projected_bbox[2].x = ((sz - m_projected_bbox[2].width) >> 1);
	m_projected_bbox[2].y = ((sz - m_projected_bbox[2].height) >> 1);
	cv::Mat zx_roi = proj_im[2](m_projected_bbox[2]);

	for (int i = 0; i < m_xyz_data.size(); ++i)
	{
		int zx_u = std::round(m_proj_k[2]*(m_xyz_data[i][2] + z_length / 2.0));
		int zx_v = std::round(m_proj_k[2]*(m_xyz_data[i][0] + x_length / 2.0));

		if (zx_u < 0)	zx_u = 0;
		else if (zx_u >= m_projected_bbox[2].width)		zx_u = m_projected_bbox[2].width - 1;

		if (zx_v < 0)	zx_v = 0;
		else if (zx_v >= m_projected_bbox[2].height)	zx_v = m_projected_bbox[2].height - 1;

		float norm_depth = (-m_xyz_data[i][1] + y_length / 2.0) / y_length;	// normalize 0-1
		if (norm_depth<0)
		{
			norm_depth = 0.0f;
		}
		if (zx_roi.at<float>(zx_v, zx_u) > norm_depth)		// set the nearest point
		{
			zx_roi.at<float>(zx_v, zx_u) = norm_depth;
		}
	}
	for (int i = 0; i < m_gt_xyz_data.size(); ++i)
	{
		proj_uv[2][i].x = std::round(m_proj_k[2]*(m_gt_xyz_data[i][2] + z_length / 2.0)) + m_projected_bbox[2].x;
		proj_uv[2][i].y = std::round(m_proj_k[2]*(m_gt_xyz_data[i][0] + x_length / 2.0)) + m_projected_bbox[2].y;

		if (proj_uv[2][i].x < 0)	proj_uv[2][i].x = 0;
		else if (proj_uv[2][i].x >= sz)	proj_uv[2][i].x = sz - 1;

		if (proj_uv[2][i].y < 0)	proj_uv[2][i].y = 0;
		else if (proj_uv[2][i].y >= sz)	proj_uv[2][i].y = sz - 1;
	}

	clock_t t0 = clock();

	cv::medianBlur(proj_im[0], proj_im[0], 5);
	cv::medianBlur(proj_im[1], proj_im[1], 5);
	cv::medianBlur(proj_im[2], proj_im[2], 5);

	return true;
}

void BoundingBox::jacobbi(const Eigen::Matrix3f& input_mat, Eigen::Matrix3f& v, double* pArray)
{
	int p, q, j, ind, n;
	double dsqr, d1, d2, thr, dv1, dv2, dv3, dmu, dga, st, ct;
	double eps = 0.00000001;
	int* iZ; //add 2002.8.27

	Eigen::Matrix3f CA(input_mat);
	n = 3;

	//add 2002.8.27
	iZ = new int[n];

	for (p = 0; p < n; p++)
	for (q = 0; q < n; q++)
		v(p,q) = (p == q) ? 1.0 : 0;

	dsqr = 0;
	for (p = 1; p < n; p++)
	for (q = 0; q < p; q++)
		dsqr += 2 * CA(p, q) * CA(p, q);
	d1 = sqrt(dsqr);
	d2 = eps / n * d1;
	thr = d1;
	ind = 0;
	do {
		thr = thr / n;
		while (!ind) {
			for (q = 1; q < n; q++)
			for (p = 0; p < q; p++)
			if (fabs(CA(p, q)) >= thr) {
				ind = 1;
				dv1 = CA(p, p);
				dv2 = CA(p, q);
				dv3 = CA(q, q);
				dmu = 0.5 * (dv1 - dv3);
				double dls = sqrt(dv2 * dv2 + dmu * dmu);
				if (fabs(dmu) < 0.00000000001) dga = -1;
				//if ( dmu == 0.0 ) dga = -1.0 ;
				else dga = (dmu < 0) ? (dv2 / dls) : (-dv2 / dls);
				st = dga / sqrt(2 * (1 + sqrt(1 - dga * dga)));
				ct = sqrt(1 - st * st);
				for (int l = 0; l < n; l++) {
					dsqr = v(l, p) * ct - v(l, q) * st;
					v(l, q) = v(l, p) * st + v(l, q) * ct;
					v(l, p) = dsqr;
				}
				for (int l = 0; l < n; l++) {
					CA(p, l) = CA(l, p);
					CA(q, l) = CA(l, q);
				}
				CA(p,p) = dv1 * ct * ct + dv3 * st * st - 2 * dv2 * st * ct;
				CA(q,q) = dv1 * st * st + dv3 * ct * ct + 2 * dv2 * st * ct;
				CA(p,q) = CA(q,p) = 0.0;
			}
			if (ind) ind = 0;
			else break;
		}
	} while (thr > d2);
	for (int l = 0; l < n; l++) {
		pArray[l] = CA(l,l);
		iZ[l] = l;
	}
	double dTemp;
	int i, k;

	for (i = 0; i < n; i++){
		//dmax = pArray[i];
		for (j = i + 1; j < n; j++){
			if (pArray[i] < pArray[j]){
				dTemp = pArray[i];
				pArray[i] = pArray[j];
				pArray[j] = dTemp;
				k = iZ[i];
				iZ[i] = iZ[j];
				iZ[j] = k;
			}
		}
	}
	CA = v;

	for (j = 0; j < n; j++)
		for (i = 0; i < n; i++)
			v(i,j) = CA(i,iZ[j]);

	delete[] iZ;
}

void BoundingBox::get_project_points(float* xyz_data, int data_num, float* xy_data, float* yz_data, float* zx_data)
{
	Eigen::Matrix4f inv_relative_trans = relative_xform.inverse();

	for (int i = 0; i < data_num; i += 3)
	{
		Eigen::Vector4f cur_xyz_data(xyz_data[i], xyz_data[i + 1], xyz_data[i + 2], 1.0f);
		if (fabs(xyz_data[i + 2]) < FLT_EPSILON)	// depth == 0.0 then skip
		{
			continue;
		}

		cur_xyz_data = inv_relative_trans * cur_xyz_data;

		Eigen::Vector4f xy_proj(cur_xyz_data);
		xy_proj[2] = -z_length / 2.0;
		xy_proj = relative_xform * xy_proj;

		Eigen::Vector4f yz_proj(cur_xyz_data);
		yz_proj[0] = x_length / 2.0;
		yz_proj, relative_xform * yz_proj;

		Eigen::Vector4f zx_proj(cur_xyz_data);
		zx_proj[1] = y_length / 2.0;
		zx_proj = relative_xform * zx_proj;

		for (int j = 0; j < 3; j++)
		{
			xy_data[i + j] = xy_proj[j];
			yz_data[i + j] = yz_proj[j];
			zx_data[i + j] = zx_proj[j];
		}
	}
}

Eigen::Vector3f BoundingBox::get_yaw_pitch_roll()
{
	Eigen::Vector3f angles(0.0, 0.0, 0.0);	// yaw-alpha, pitch-beta, roll-garma

	angles[0] = atan2(-relative_xform(2, 0), sqrt(pow(relative_xform(0, 0), 2) + pow(relative_xform(1, 0), 2)));
	angles[1] = atan2(relative_xform(1, 0), relative_xform(0, 0));
	angles[2] = atan2(relative_xform(2, 1), relative_xform(2, 2));

	return angles;
}
