#ifndef PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H
#define PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H

#include <opencv2/opencv.hpp>
#include "gpu_utilities.cuh"
#include "common/utilities.hpp"
#include "gpu_img_transform.cuh"


struct StreamInfo {
    long size_bytes;
    double size;
    double size_block;
    long rows;
    StreamInfo(const StreamInfo &streams_info, int nb_streams, long cols, ConvolutionMatrixProperties &conv_prop);
    StreamInfo(double size, long rows);
};
class GpuImgTransformStream {
    static void initMemory(cv::Mat &m_in, ConvMatrixPointers &dev_convolution,
                           Pointers &host, long size, int conv_mat_length);
    static void initStreamAndDevMem(StreamInfo &per_stream_info, int nb_streams, cudaStream_t *streams,
                                    RgbPointers *dev_rgbs, unsigned char *host_rgb_in);
    static void swapStreamMem(StreamInfo &per_stream_info, int nb_streams, RgbPointers *dev_rgbs);
    static void freeMemory(RgbPointers *dev_rgbs, ConvMatrixPointers &dev_convolution, Pointers &host, int nb_streams);
public:
    static int execute(cv::Mat &img_in, cv::Mat &img_out, GpuUtilExecutionInfo &info);
    static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, GpuUtilExecutionInfo &info);
};

#endif //PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H
