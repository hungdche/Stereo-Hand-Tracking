#ifndef __HEATMAP_FUSION__
#define __HEATMAP_FUSION__

#include "depth_projector.h"
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

class HeatmapFuser {
private:
    // Set at initialization (parameters of the camera)
    int m_x_res, m_y_res;
    double m_focal_length;
    int m_out_size, m_heat_size;

    // 
    std::array<double, 3> bbox_x;
    std::array<double, 3> bbox_y;
    std::array<double, 3> bbox_w;
    std::array<double, 3> bbox_h;

    // std::vector<std::vectoor
    

}


#endif // __HEATMAP_FUSION__
