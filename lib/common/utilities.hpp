//
// Created by tsky on 06/03/2021.
//

#ifndef PROJET_CUDA_CONSTANTS_H
#define PROJET_CUDA_CONSTANTS_H

enum EffectStyle {
    BOXBLUR, GAUSSIANBLUR, GAUSSIANBLUR5, EMBOSS, EMBOSS5, SHARPEN
};

struct ConvolutionMatrixProperties {
    int size;
    int divisor;
    int start_index;
};

static char ARR_BOXBLUR[3*3] = { 1, 1, 1,
                              1, 1, 1,
                              1, 1, 1
};
static char ARR_GAUSSIANBLUR[3*3] = {  1, 2, 1,
                                    2, 4, 2,
                                    1, 2, 1
};
static char ARR_GAUSSIANBLUR5[5*5] = { 1, 4, 6, 4, 1,
                                    4, 16, 24, 16, 4,
                                    6, 24, 36, 24, 6,
                                    4, 16, 24, 16, 4,
                                    1, 4, 6, 4, 1
};
static char ARR_EMBOSS[3*3] = { -2, -1, 0,
                             -1, 1, 1,
                             0, 1, 2
};
static char ARR_EMBOSS5[5*5] = { 1, 0, 0, 0, 0,
                              0, 1, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 0, -1, 0,
                              0, 0, 0, 0, -1
};
static char ARR_SHARPEN[3*3] = { 0, -5, 0,
                              -5, 3, -5,
                              0, -5, 0
};

void set_convolution_properties(ConvolutionMatrixProperties &mat_def_properties, EffectStyle style);
void swapPointers( unsigned char **ptr1, unsigned char **ptr2 );
void copyReverse(char *desti, EffectStyle style, int length);

#endif //PROJET_CUDA_CONSTANTS_H
