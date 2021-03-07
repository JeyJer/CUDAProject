#include "utilities.hpp"
void copyReverse(char *desti, char *source, int length);

void set_convolution_properties(ConvolutionMatrixProperties &mat_def_properties, EffectStyle style){
    switch( style){
        case EMBOSS5:
            mat_def_properties.size = 5;
            mat_def_properties.divisor = 1;
            break;
        case EMBOSS:
            mat_def_properties.size = 3;
            mat_def_properties.divisor = 1;
            break;
        case GAUSSIANBLUR5:
            mat_def_properties.size = 5;
            mat_def_properties.divisor = 256;
            break;
        case GAUSSIANBLUR:
            mat_def_properties.size = 3;
            mat_def_properties.divisor = 16;
            break;
        case SHARPEN:
            mat_def_properties.size = 3;
            mat_def_properties.divisor = -17;
            break;
        default:
            mat_def_properties.size = 3;
            mat_def_properties.divisor = 9;
    }

    mat_def_properties.start_index = -(mat_def_properties.size - 1 ) / 2;
}

void copyReverse(char *desti, char *source, int length) {
    for( int i =0, j = length - 1; i < length; i++, j--) desti[i] = source[j];
}
void copyReverse(char *desti, EffectStyle style, int length){
    switch( style){
        case EMBOSS5:
            copyReverse(desti, ARR_EMBOSS5, length);
            break;
        case EMBOSS:
            copyReverse(desti, ARR_EMBOSS, length);
            break;
        case GAUSSIANBLUR5:
            copyReverse(desti, ARR_GAUSSIANBLUR5, length);
            break;
        case GAUSSIANBLUR:
            copyReverse(desti, ARR_GAUSSIANBLUR, length);
            break;
        case SHARPEN:
            copyReverse(desti, ARR_SHARPEN, length);
            break;
        default:
            copyReverse(desti, ARR_BOXBLUR, length);
    }

}
void swapPointers( unsigned char **ptr1, unsigned char **ptr2 )
{
    unsigned char* invertion_ptr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = invertion_ptr;
}