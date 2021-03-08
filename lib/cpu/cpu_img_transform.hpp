//
// Created by tsky on 08/03/2021.
//

#ifndef PROJET_CUDA_CPU_IMG_TRANSFORM_H
#define PROJET_CUDA_CPU_IMG_TRANSFORM_H

#include <opencv2/opencv.hpp>
#include "cpu_utilities.hpp"


class CpuImgTransform {
private:
    static void transform_img(unsigned char* input, unsigned char* output, std::size_t nb_cols, std::size_t nb_rows,
                         char *conv_mat, ConvolutionMatrixProperties &conv_mat_properties);
    static void initMemory(cv::Mat &m_in, Pointers &host, long size, int conv_mat_length);
    static void freeMemory(Pointers &host);
public:
    static int execute(cv::Mat &img_in, cv::Mat &img_out, CpuUtilExecutionInfo &info);
};

#endif //PROJET_CUDA_CPU_IMG_TRANSFORM_H
