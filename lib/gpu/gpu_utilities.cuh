#ifndef PROJET_CUDA_MENU_LIB_H
#define PROJET_CUDA_MENU_LIB_H

#include "utilities.hpp"
#include <vector>
#include <string>

struct GpuUtilExecutionInfo {
    char *conv_matrix;
    ConvolutionMatrixProperties conv_properties;
    int nb_pass;
    dim3 block;
    int nb_streams;
};

class GpuUtilMenuSelection {
    static void printOptionSelection(std::string &img_in_path, std::string &img_out_path, GpuUtilMenuSelection &menuChoice);
    static void presavedParameters(std::string &img_in_path, std::string &img_out_path, GpuUtilMenuSelection &menuChoice );

public:
    bool use_shared;
    int nb_stream;
    struct {
        int dimX;
        int dimY;
    } block;
    std::vector<EffectStyle> enabled_filters;
    std::vector<int> nb_pass;

    static int initParameters(std::string &img_in_path, std::string &img_out_path,
                              GpuUtilMenuSelection &menuChoice,
                              int argc , char **argv );
};

#endif //PROJET_CUDA_MENU_LIB_H
