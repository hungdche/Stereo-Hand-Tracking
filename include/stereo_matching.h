#ifndef __STEREO_MATCHING_H__
#define __STEREO_MATCHING_H__

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stereo/stereo.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

class StereoMatcher {
private:
    // StereoBM
    int _m_num_disp   = 5;
    int _m_block_size = 34;
    int _m_pref_type  = 1;
    int _m_pref_size  = 0;
    int _m_pref_cap   = 62;
    int _m_tex_thresh = 0;
    int _m_uniq_ratio = 4;
    int _m_spec_range = 100;
    int _m_spec_wsize = 8;
    int _m_disp12_max = 74;
    int _m_min_disp   = 42;

    // WLS Filter
    int _m_wls_sigma = 13;
    int _m_wls_lambda = 3;

    // Foreground Guided Filter
    int _m_foreg_radius = 12;
    int _m_foreg_eps = 6;

    // Color Guided Filter
    int _m_color_radius = 19;
    int _m_color_eps = 13;

    // matchers
    cv::Ptr<cv::StereoBM> left_matcher;
    cv::Ptr<cv::StereoMatcher> right_matcher;

    // filters
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    cv::Ptr<cv::ximgproc::GuidedFilter> guided_filter; 

public:
    StereoMatcher(int n_disparity = 21, int block_size = 5);
    cv::Mat compute(const cv::Mat &left_color, const cv::Mat &right_color, const cv::Mat &foreground);
    void debug(const cv::Mat &left_color, const cv::Mat &right_color, const cv::Mat &foreground);

private:
    // private helper functions
    cv::Mat compute_disparity(const cv::Mat& left_gray, const cv::Mat& right_gray, const cv::Mat& left_color);
    cv::Mat filter_guided(const cv::Mat guider, const cv::Mat input, float radius, float reg);
};



#endif // __STEREO_MATCHING_H__