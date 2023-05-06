#include "stereo_matching.h"

StereoMatcher::StereoMatcher(int n_disparity, int block_size) 
{
    left_matcher = cv::StereoBM::create();  

    left_matcher->setNumDisparities((_m_num_disp + 1) * 16);
    left_matcher->setBlockSize(_m_block_size*2+5);
    left_matcher->setPreFilterType(_m_pref_type);
    left_matcher->setPreFilterSize(_m_pref_size*2+5);
    left_matcher->setPreFilterCap(_m_pref_cap+1);
    left_matcher->setMinDisparity(_m_min_disp-25);
    left_matcher->setTextureThreshold(_m_tex_thresh);
    left_matcher->setUniquenessRatio(_m_uniq_ratio);
    left_matcher->setSpeckleRange(_m_spec_range);
    left_matcher->setSpeckleWindowSize(_m_spec_wsize*2+1);
    left_matcher->setDisp12MaxDiff(_m_disp12_max-50);

    right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
    wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);  

    wls_filter->setLambda(_m_wls_lambda*1000);
    wls_filter->setSigmaColor((double)_m_wls_sigma / 10.0);
}

cv::Mat StereoMatcher::compute_disparity(const cv::Mat& left_gray, const cv::Mat& right_gray, const cv::Mat& left_color) {
    cv::Mat left_disp, right_disp, disparity;
    left_matcher->compute(left_gray,right_gray,left_disp);
    right_matcher->compute(right_gray, left_gray, right_disp);
    
    left_disp.convertTo(left_disp,CV_32F, 1.0);
    right_disp.convertTo(right_disp,CV_32F, 1.0);

    // left_disp = (left_disp/16.0f - (float)(_m_min_disp-25))/((float)(_m_num_disp + 1) * 16);
    // right_disp = (right_disp/16.0f - (float)(_m_min_disp-25))/((float)(_m_num_disp + 1) * 16);
    
    wls_filter->filter(left_disp,left_color,disparity,right_disp);
    
    return disparity;
}

cv::Mat StereoMatcher::filter_guided(const cv::Mat guider, const cv::Mat input, float radius, float reg) {
    cv::Mat filtered;
    guided_filter = cv::ximgproc::createGuidedFilter(guider, radius+1, pow((double)(reg+1)/100.0,2));
    guided_filter->filter(input, filtered);
    return filtered;
}

cv::Mat StereoMatcher::compute(const cv::Mat &left_color, const cv::Mat &right_color, const cv::Mat &foreground) {
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_color, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_color, right_gray, cv::COLOR_BGR2GRAY);

    cv::Mat disparity, filtered, final_depth;
    disparity = compute_disparity(left_gray, right_gray, left_color);
    cv::ximgproc::getDisparityVis(disparity,disparity,1);
    filtered = filter_guided(foreground, disparity, _m_foreg_radius, _m_foreg_eps);
    final_depth = filter_guided(left_color, filtered, _m_color_radius, _m_color_radius);
    final_depth.convertTo(final_depth,CV_32F, 1.0);
    return final_depth;
}

#pragma region DEBUG_VAR
// stereo_bm
int debug_num_disp   = 9;
int debug_block_size = 2;
int debug_pref_type  = 1;
int debug_pref_size  = 25;
int debug_pref_cap   = 62;
int debug_tex_thresh = 0;
int debug_uniq_ratio = 4;
int debug_spec_range = 100;
int debug_spec_wsize = 9;
int debug_disp12_max = 50;
int debug_min_disp   = 25;

// wls_filter
int debug_wls_sigma = 20;
int debug_wls_lambda = 8; 

// foreground_filter 
int debug_foreg_radius = 10;
int debug_foreg_eps = 8;

// color_filter
int debug_color_radius = 19;
int debug_color_eps = 13;
#pragma endregion

#pragma region SLIDER
static void _num_disp_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setNumDisparities((debug_num_disp + 1) * 16);
}
static void _block_size_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setBlockSize(debug_block_size*2+5); 
}
static void _pref_type_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setPreFilterType(debug_pref_type); 
}
static void _pref_size_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setPreFilterSize(debug_pref_size*2+5); 
}
static void _pref_cap_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setPreFilterCap(debug_pref_cap+1);
}
static void _tex_thresh_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setTextureThreshold(debug_tex_thresh); 
}
static void _unq_ratio_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setUniquenessRatio(debug_uniq_ratio); 
}
static void _spec_range_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setSpeckleRange(debug_spec_range); 
}
static void _spec_wsize_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setSpeckleWindowSize(debug_spec_wsize*2+1);
}
static void _disp12_max_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setDisp12MaxDiff(debug_disp12_max-50); 
}
static void _min_disp_bar( int, void* input) { 
    cv::Ptr<cv::StereoBM> left_matcher = *((cv::Ptr<cv::StereoBM>*)input);
    left_matcher->setMinDisparity(debug_min_disp-25); 
}
static void _wls_lambda_bar( int, void* input) { 
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    wls_filter = *((cv::Ptr<cv::ximgproc::DisparityWLSFilter>*)input);
    wls_filter->setLambda(debug_wls_lambda*1000);  
}
static void _wls_sigma_bar( int, void* input) { 
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
    wls_filter = *((cv::Ptr<cv::ximgproc::DisparityWLSFilter>*)input);
    wls_filter->setSigmaColor((double)debug_wls_sigma / 10.0); 
}
#pragma endregion

cv::Mat StereoMatcher::debug(const cv::Mat &left_color, const cv::Mat &right_color, const cv::Mat &foreground) {
    // initial disparity window
    cv::namedWindow("disparity",cv::WINDOW_NORMAL);
    cv::resizeWindow("disparity",600,600);

    // Creating trackbars to dynamically update the StereoBM parameters
    cv::createTrackbar("numDisparities", "disparity", &debug_num_disp, 18, _num_disp_bar, &left_matcher);
    cv::createTrackbar("blockSize", "disparity", &debug_block_size, 50, _block_size_bar, &left_matcher);
    cv::createTrackbar("preFilterType", "disparity", &debug_pref_type, 1, _pref_type_bar, &left_matcher);
    cv::createTrackbar("preFilterSize", "disparity", &debug_pref_size, 25, _pref_size_bar, &left_matcher);
    cv::createTrackbar("preFilterCap", "disparity", &debug_pref_cap, 62, _pref_cap_bar, &left_matcher);
    cv::createTrackbar("textureThreshold", "disparity", &debug_tex_thresh, 100, _tex_thresh_bar, &left_matcher);
    cv::createTrackbar("uniquenessRatio", "disparity", &debug_uniq_ratio, 100, _unq_ratio_bar, &left_matcher);
    cv::createTrackbar("speckleRange", "disparity", &debug_spec_range, 100, _spec_range_bar, &left_matcher);
    cv::createTrackbar("speckleWindowSize", "disparity", &debug_spec_wsize, 25, _spec_wsize_bar, &left_matcher);
    cv::createTrackbar("disp12MaxDiff", "disparity", &debug_disp12_max, 100, _disp12_max_bar, &left_matcher);
    cv::createTrackbar("minDisparity", "disparity", &debug_min_disp, 50, _min_disp_bar, &left_matcher);

    // Creating trackbars to dynamically update WLS Filter params
    cv::createTrackbar("wlsSigmaColor", "disparity", &debug_wls_sigma, 20, _wls_sigma_bar, &wls_filter);
    cv::createTrackbar("wlsLambda", "disparity", &debug_wls_lambda, 8, _wls_lambda_bar, &wls_filter);

    // downscale the images
    // cv::Mat left_scaled, right_scaled;
    // cv::resize(left_color ,left_scaled ,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
    // cv::resize(right_color,right_scaled,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);

    // Converting images to grayscale
    cv::Mat left_gray, right_gray;
    cv::cvtColor(left_color, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right_color, right_gray, cv::COLOR_BGR2GRAY);

    // Obtaining ideal params for initial disparity
    cv::Mat raw_disp;
    while (cv::getWindowProperty("disparity", cv::WND_PROP_AUTOSIZE) >= 0) {
        right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
        wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);  
        wls_filter->setLambda(debug_wls_lambda*1000);
        wls_filter->setSigmaColor((double)debug_wls_sigma / 10.0);
        cv::Mat disparity = compute_disparity(left_gray, right_gray, left_color);
        cv::ximgproc::getDisparityVis(disparity,raw_disp,1);
        cv::imshow("disparity",raw_disp);
        if (cv::waitKey(20) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }
    // load all debug params 
    _m_num_disp   = debug_num_disp;
    _m_block_size = debug_block_size;
    _m_pref_type  = debug_pref_type;
    _m_pref_size  = debug_pref_size;
    _m_pref_cap   = debug_pref_cap; 
    _m_min_disp   = debug_min_disp;
    _m_tex_thresh = debug_tex_thresh; 
    _m_uniq_ratio = debug_uniq_ratio; 
    _m_spec_range = debug_spec_range;
    _m_spec_wsize = debug_spec_wsize;
    _m_disp12_max = debug_disp12_max; 
    _m_wls_lambda = debug_wls_lambda;
    _m_wls_sigma  = debug_wls_sigma;

    // showing the confidence of the latest WLS Filter call
    cv::namedWindow("confidence",cv::WINDOW_NORMAL);
    cv::Mat conf = wls_filter->getConfidenceMap();
    cv::imshow("confidence",conf);
    while (cv::getWindowProperty("confidence", cv::WND_PROP_AUTOSIZE) >= 0) {
        // cv::imshow("confidence",conf);
        if (cv::waitKey(20) == 27){
            cv::destroyAllWindows();
            break;
        }
    }

    // showing the foreground filter window
    cv::namedWindow("foreground_filter",cv::WINDOW_NORMAL);
    cv::resizeWindow("foreground_filter",600,600);

    // trackbars for the foreground filter
    cv::createTrackbar("radius", "foreground_filter", &debug_foreg_radius, 50);
    cv::createTrackbar("eps", "foreground_filter", &debug_foreg_eps, 100);

    // obtaining ideal param for foreground filter
    cv::Mat filtered;
    while (cv::getWindowProperty("foreground_filter", cv::WND_PROP_AUTOSIZE) >= 0) {
        filtered = filter_guided(foreground, raw_disp, debug_foreg_radius, debug_foreg_eps);

        cv::imshow("foreground_filter",filtered);
        if (cv::waitKey(20) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }

    // load foreground params
    _m_foreg_radius = debug_foreg_radius;
    _m_foreg_eps = debug_foreg_eps;

    // TODO: Perform the matching cost calculating process here

    // showing color guilded filter window
    cv::namedWindow("color_filter", cv::WINDOW_NORMAL);
    cv::resizeWindow("color_filter",600,600);

    // trackbars for color guided filter
    cv::createTrackbar("radius", "color_filter", &debug_color_radius, 50);
    cv::createTrackbar("eps", "color_filter", &debug_color_eps, 100);

    // obtaining ideal color guided filter params
    cv::Mat final_depth;
    while (cv::getWindowProperty("color_filter", cv::WND_PROP_AUTOSIZE) >= 0) {
        final_depth = filter_guided(left_color, filtered, debug_color_radius, debug_color_eps);
        cv::imshow("color_filter",final_depth);
        if (cv::waitKey(20) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }

    // load color filter params
    _m_color_radius = debug_color_radius;
    _m_color_eps = debug_color_eps;

    // dump params
    std::cout << "================STEREOBM PARAM================\n";
    std::cout << "numDisparities: " << _m_num_disp << std::endl;
    std::cout << "blockSize: " << _m_block_size << std::endl;
    std::cout << "preFilterType: " << _m_pref_type << std::endl;
    std::cout << "preFilterSize: " << _m_pref_size << std::endl;
    std::cout << "preFilterCap: " << _m_pref_cap << std::endl;
    std::cout << "textureThreshold: " << _m_tex_thresh << std::endl;
    std::cout << "uniquenessRatio: " << _m_uniq_ratio << std::endl;
    std::cout << "speckleRange: " << _m_spec_range << std::endl;
    std::cout << "speckleWindowSize: " << _m_spec_wsize << std::endl;
    std::cout << "disp12MaxDiff: " << _m_disp12_max << std::endl;
    std::cout << "minDisparity: " << _m_min_disp << std::endl;
    std::cout << "==============================================\n";

    std::cout << "==================WLS PARAM===================\n";
    std::cout << "wlsSigmaColor: " << _m_wls_sigma << std::endl;
    std::cout << "wlsLambda: " << _m_wls_lambda << std::endl;
    std::cout << "==============================================\n";

    std::cout << "===============FOREGROUND PARAM===============\n";
    std::cout << "radius: " << _m_foreg_radius << std::endl;
    std::cout << "eps: " << _m_foreg_eps << std::endl;
    std::cout << "==============================================\n";

    std::cout << "=================COLOR PARAM==================\n";
    std::cout << "radius: " << _m_color_radius << std::endl;
    std::cout << "eps: " << _m_color_eps << std::endl;
    std::cout << "==============================================\n";
    
    // final_depth.convertTo(final_depth,CV_32F, 1.0);
    return final_depth;
}