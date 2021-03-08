#ifndef PROJET_CUDA_GPU_IMG_TRANSFORM_CUH
#define PROJET_CUDA_GPU_IMG_TRANSFORM_CUH

#include <opencv2/opencv.hpp>
#include "gpu_utilities.cuh"
#include "common/utilities.hpp"

class GpuImgTransform {
    static void initMemory(cv::Mat &m_in, Pointers &dev, Pointers &host, long size, int conv_mat_length);
    static void freeMemory(Pointers &dev, Pointers &host);

public:
    static int execute(cv::Mat &img_in, cv::Mat &img_out, GpuUtilExecutionInfo &info);
    static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, GpuUtilExecutionInfo &info);
};
#endif //PROJET_CUDA_GPU_IMG_TRANSFORM_CUH
