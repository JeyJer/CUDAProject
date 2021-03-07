//
// Created by tsky on 07/03/2021.
//

#ifndef PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H
#define PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H

#include <opencv2/opencv.hpp>
#include "common/menu_lib.hpp"
#include "common/utilities.hpp"
#include "gpu_img_transform.cuh"

struct StreamProperty {
    long units;
    long bytes;
    StreamProperty(long u, long b): units(u), bytes(b){}
};
class GpuImgTransFormStream {
public:
    static int execute(cv::Mat &img_in, cv::Mat &img_out, ExecutionInfo &info);
    static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, ExecutionInfo &info);
};


#endif //PROJET_CUDA_GPUIMGTRANSFORMSTREAM_H
