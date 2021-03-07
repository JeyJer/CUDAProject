#ifndef PROJET_CUDA_CPU_CONVOLUTION_TRANSFORM_HPP
#define PROJET_CUDA_CPU_CONVOLUTION_TRANSFORM_HPP
#include "common/convmatrix_lib.hpp"
#include "common/menu_lib.hpp"

using namespace std;

class CpuConvolutionTransform {
public:
    CpuConvolutionTransform(char *conv_mat, ConvolutionMatrixProperties &conv_prop, unsigned char *input, int nb_cols, int nb_rows);
    virtual ~CpuConvolutionTransform();
    unsigned char *output;
private:
    char *conv_mat;
    unsigned char *input;
    int nb_cols;
    int nb_rows;
    ConvolutionMatrixProperties conv_prop;
public:
    void transformPixel(int nb_cols, int nb_rows);
    void saveOutput2Input();
};

#endif //PROJET_CUDA_CPU_CONVOLUTION_TRANSFORM_HPP
