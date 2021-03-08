//
// Created by tsky on 08/03/2021.
//

#ifndef PROJET_CUDA_CPU_UTILITIES_H
#define PROJET_CUDA_CPU_UTILITIES_H

#include "utilities.hpp"
#include <vector>
#include <string>

struct CpuUtilPointers {
    char* conv_matrix;
    ConvolutionMatrixProperties *conv_prop;
    unsigned char* rgb_in;
    unsigned char* rgb_out;
};
struct CpuUtilExecutionInfo {
    char *conv_matrix;
    ConvolutionMatrixProperties conv_properties;
    int nb_pass;
};

class CpuUtilMenuSelection {
    static void printOptionSelection(std::string &img_in_path, std::string &img_out_path,
            CpuUtilMenuSelection &menuChoice);
    static void presavedParameters(std::string &img_in_path, std::string &img_out_path,
            CpuUtilMenuSelection &menuChoice );
public:
    std::vector<EffectStyle> enabled_filters;
    std::vector<int> nb_pass;

    static int initParameters(std::string &img_in_path, std::string &img_out_path,
                              CpuUtilMenuSelection &menuChoice,
                              int argc , char **argv );
};

#endif //PROJET_CUDA_CPU_UTILITIES_H
