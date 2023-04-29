#include "stereo_matching.h"

StereoMatcher::StereoMatcher(int n_disparity, int block_size) 
{
    left_matcher = cv::StereoBM::create();
    right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
    wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
}



cv::Mat StereoMatcher::compute(const cv::Mat &left, const cv::Mat &right) {
    cv::Mat gray_left, gray_right;
    cv::cvtColor(left, gray_left, CV_BGR2GRAY);
    cv::cvtColor(right, gray_right, CV_BGR2GRAY);
    cv::Mat ldisp, rdisp, filtered_disp; 
    left_matcher->compute(gray_left, gray_right, ldisp);
    right_matcher->compute(gray_right, gray_left, rdisp);
    wls_filter->filter(ldisp,left,filtered_disp,rdisp);

    ldisp.convertTo(ldisp, CV_8U);

    // cv::Mat norm; 
    // cv::normalize(ldisp, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Point p_min, p_max;
    cv::minMaxLoc(ldisp, NULL, NULL, &p_min, &p_max);
    std::cout << "[DISP]: min " << ldisp.at<float>(p_min.y, p_min.x) << 
                        " max " << ldisp.at<float>(p_max.y, p_max.x) << std::endl;
    return ldisp;
}


// variables for StereoBM
int _m_num_disp   = 8;
int _m_block_size = 5;
int _m_pref_type  = 1;
int _m_pref_size  = 1;
int _m_pref_cap   = 31;
int _m_min_disp   = 0;
int _m_tex_thresh = 10;
int _m_uniq_ratio = 15;
int _m_spec_range = 0;
int _m_spec_wsize = 0;
int _m_disp12_max = -1;

void _num_disp_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    _m_num_disp *= 16; left_matcher->setNumDisparities(_m_num_disp); 
}
void _block_size_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    _m_block_size = _m_block_size*2+5; left_matcher->setBlockSize(_m_block_size);
}
void _pref_type_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setPreFilterType(_m_pref_type); 
}
void _pref_size_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    _m_pref_size = _m_pref_size*2+5; left_matcher->setPreFilterSize(_m_pref_size); 
}
void _pref_cap_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setPreFilterCap(_m_pref_cap); 
}
void _tex_thresh_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setTextureThreshold(_m_tex_thresh); 
}
void _unq_ratio_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setUniquenessRatio(_m_uniq_ratio); 
}
void _spec_range_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setSpeckleRange(_m_spec_range); 
}
void _spec_wsize_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    _m_spec_wsize *= 2; left_matcher->setSpeckleWindowSize(_m_spec_wsize);
}
void _disp12_max_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setDisp12MaxDiff(_m_disp12_max); 
}
void _min_disp_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = (cv::Ptr<cv::StereoBM>)input;
    left_matcher->setMinDisparity(_m_min_disp); 
}

void StereoMatcher::debug(const cv::Mat &left_color, const cv::Mat &right_color, const cv::Mat &foreground) {
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity",600,600);
    
    // Creating trackbars to dynamically update the StereoBM parameters
    cv::createTrackbar("numDisparities", "disparity", &_m_num_disp, 18, _num_disp_bar, left_matcher);
    cv::createTrackbar("blockSize", "disparity", &_m_block_size, 50, _block_size_bar, left_matcher);
    cv::createTrackbar("preFilterType", "disparity", &_m_pref_type, 1, _pref_type_bar, left_matcher);
    cv::createTrackbar("preFilterSize", "disparity", &_m_pref_size, 25, _pref_size_bar, left_matcher);
    cv::createTrackbar("preFilterCap", "disparity", &_m_pref_cap, 62, _pref_cap_bar, left_matcher);
    cv::createTrackbar("textureThreshold", "disparity", &_m_tex_thresh, 100, _tex_thresh_bar, left_matcher);
    cv::createTrackbar("uniquenessRatio", "disparity", &_m_uniq_ratio, 100, _unq_ratio_bar, left_matcher);
    cv::createTrackbar("speckleRange", "disparity", &_m_spec_range, 100, _spec_range_bar, left_matcher);
    cv::createTrackbar("speckleWindowSize", "disparity", &_m_spec_wsize, 25, _spec_wsize_bar, left_matcher);
    cv::createTrackbar("disp12MaxDiff", "disparity", &_m_disp12_max, 25, _disp12_max_bar, left_matcher);
    cv::createTrackbar("minDisparity", "disparity", &_m_min_disp, 25, _min_disp_bar, left_matcher);
    while(true) {
        
    }
}

