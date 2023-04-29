#ifndef __STEREO_MATCHING_H__
#define __STEREO_MATCHING_H__

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/stereo/stereo.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// struct StereoParams {
//     int _m_num_disp;   int _m_block_size;
//     int _m_pref_type;  int _m_pref_size; 
//     int _m_pref_cap;   int _m_min_disp;  
//     int _m_tex_thresh; int _m_uniq_ratio;
//     int _m_spec_range; int _m_spec_wsize;
//     int _m_disp12_max; 


// };

class StereoMatcher {
private:
    // variables for StereoBM
    // StereoParams* params;

    // matchers
    cv::Ptr<cv::StereoBM> left_matcher;
    cv::Ptr<cv::StereoMatcher> right_matcher;

    // filters
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    cv::Ptr<cv::ximgproc::GuidedFilter> guided_filter; 

public:
    StereoMatcher(int n_disparity, int block_size = 5);
    cv::Mat compute(const cv::Mat &left, const cv::Mat &right);
    void debug(const cv::Mat &left_color, const cv::Mat &right_color, const cv::Mat &foreground);

    
};



#endif // __STEREO_MATCHING_H__