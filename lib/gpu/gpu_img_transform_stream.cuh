#ifndef PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H
#define PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H

#include <opencv2/opencv.hpp>
#include "gpu_utilities.cuh"
#include "common/utilities.hpp"
#include "gpu_img_transform.cuh"


struct StreamsInfo {
    long *sizes;
    long *effective_sizes;
    long *rows;
    StreamsInfo(long rows, long cols, int nb_streams, ConvolutionMatrixProperties &conv_prop);
};
class GpuImgTransformStream {
    static void initMemory(cv::Mat &m_in, ConvMatrixPointers &dev_convolution,
                           Pointers &host, long size, int conv_mat_length);
    static void initStreamAndDevMem(StreamsInfo &info_streams, int nb_streams, cudaStream_t *streams,
                                    RgbPointers *dev_rgbs, unsigned char *host_rgb_in);
    static void swapStreamMem(StreamsInfo &info_streams, int nb_streams, RgbPointers *dev_rgbs);
    static void freeMemory(RgbPointers *dev_rgbs, ConvMatrixPointers &dev_convolution, Pointers &host, int nb_streams);
public:
    static int execute(cv::Mat &img_in, cv::Mat &img_out, GpuUtilExecutionInfo &info);
    static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, GpuUtilExecutionInfo &info);
};

#endif //PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H
