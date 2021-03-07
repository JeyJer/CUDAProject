#ifndef PROJET_CUDA_MENU_LIB_H
#define PROJET_CUDA_MENU_LIB_H

#include "utilities.hpp"
#include <vector>
#include <string>

struct MenuSelection {
    bool use_shared;
    int nb_stream;
    struct {
        int dimX;
        int dimY;
    } block;
    std::vector<EffectStyle> enabled_filters;
    std::vector<int> nb_pass;
};

int initParameters( std::string &img_in_path, std::string &img_out_path,
                     MenuSelection &menuChoice,
                     int argc , char **argv );

#endif //PROJET_CUDA_MENU_LIB_H
