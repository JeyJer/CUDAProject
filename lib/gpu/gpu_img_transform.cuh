#ifndef PROJET_CUDA_GPU_IMG_TRANSFORM_CUH
#define PROJET_CUDA_GPU_IMG_TRANSFORM_CUH

#include <opencv2/opencv.hpp>
#include "common/menu_lib.hpp"

struct RefPointers {
    char* conv_matrix;
    ConvolutionMatrixProperties *conv_prop;
    unsigned char* rgb_in;
    unsigned char* rgb_out;
};
struct ExecutionInfo {
    char *conv_matrix;
    ConvolutionMatrixProperties conv_properties;
    int nb_pass;
    dim3 block;
};


class GpuImgTransform {
public:
    static int execute(cv::Mat &img_in, cv::Mat &img_out, ExecutionInfo &info);
    static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, ExecutionInfo &info);
};
#endif //PROJET_CUDA_GPU_IMG_TRANSFORM_CUH
