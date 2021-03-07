#include <cstdio>
#include "cpu_convolution_transform.hpp"
#include "common/convmatrix_lib.hpp"

CpuConvolutionTransform::CpuConvolutionTransform(char *conv_mat, ConvolutionMatrixProperties &conv_prop,
        unsigned char *input, int nb_cols, int nb_rows):
    input(input), nb_rows(nb_rows), nb_cols(nb_cols), output(new unsigned char[nb_cols * nb_rows * 3]),
    conv_mat(conv_mat), conv_prop(conv_prop){
}
void CpuConvolutionTransform::transformPixel( int ith_col, int jth_row) {
    if (ith_col + conv_prop.start_index  >= 0 && ith_col < nb_cols + conv_prop.start_index &&
        jth_row + conv_prop.start_index >= 0 && jth_row < nb_rows + conv_prop.start_index)
    {
        int j_local = jth_row + conv_prop.start_index;
        int i_local;

        int i_max = ith_col + conv_prop.start_index + conv_prop.size;
        int j_max = j_local + conv_prop.size;
        int rgb[3] = {0, 0, 0};
        for( int j_inc = 0; j_local < j_max; j_local++){

            i_local = i_max - conv_prop.size;
            long index =  3 * (j_local * nb_cols + i_local);
            for(  ; i_local < i_max; i_local++, j_inc++ ){
                rgb[0] += conv_mat[j_inc] * input[ index ];
                rgb[1] += conv_mat[j_inc] * input[ index + 1 ];
                rgb[2] += conv_mat[j_inc] * input[ index + 2 ];

                index += 3;
            }
        }
        for( int i = 0, j = 3 * (jth_row * nb_cols + ith_col); i < 3; i++, j++)
            output[j] = rgb[i] / conv_prop.divisor;
    }

}
void CpuConvolutionTransform::saveOutput2Input() {
    for(int i = 0; i < 3 * nb_rows * nb_cols; i++) input[i] = output[i];
}
CpuConvolutionTransform::~CpuConvolutionTransform() = default;


