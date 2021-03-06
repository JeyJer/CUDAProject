#include <cstdio>
#include "convolution_transform.hpp"

ConvolutionMatrix::ConvolutionMatrix(EffectStyle style) {
    switch( style){
        case EMBOSS5:
            set(
                    new char[5*5]{
                            1, 0, 0, 0, 0,
                            0, 1, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, -1, 0,
                            0, 0, 0, 0, -1
                    },5, 1);
            break;
        case EMBOSS:
            set(
                    new char[3*3]{
                            -2, -1, 0,
                            -1, 1, 1,
                            0, 1, 2
                    },3, 1);
            break;
        case GAUSSIANBLUR5:
            set(
                    new char[5*5]{
                            1, 4, 6, 4, 1,
                            4, 16, 24, 16, 4,
                            6, 24, 36, 24, 6,
                            4, 16, 24, 16, 4,
                            1, 4, 6, 4, 1
                    }, 5, 256);
            break;
        case GAUSSIANBLUR:
            set(
                    new char[3*3]{
                            1, 2, 1,
                            2, 4, 2,
                            1, 2, 1
                    },3, 16);
            break;
        case SHARPEN:
            set(
                    new char[3*3]{
                            0, -5, 0,
                            -5, 3, -5,
                            0, -5, 0
                    }, 3, -17);
            break;
        default:
            set(
                    new char[3*3]{
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1
                    }, 3, 9);
    }
}
void ConvolutionMatrix::set(char *mat, int size_val, int div_val) {
    int size_max = size_val * size_val;
    matrix = new char[size_max];
    for( int i = 0, j = size_max-1; i < size_max; i++, j--){
        matrix[i] = mat[j];
    }
    this->size = size_val;
    this->divisor = div_val;
    this->start_index = -(size_val - 1 )/ 2;
}


ConvolutionTransform::ConvolutionTransform(ConvolutionMatrix &convMatrix,
        unsigned char *input, int nb_cols, int nb_rows): convolutionMatrix(convMatrix), input(input),
    nb_rows(nb_rows), nb_cols(nb_cols), output(new unsigned char[nb_cols * nb_rows * 3]){}

void ConvolutionTransform::transformPixel( int ith_col, int jth_row) {
    if (ith_col > 0 && ith_col < nb_cols + convolutionMatrix.start_index &&
        jth_row > 0 && jth_row < nb_rows + convolutionMatrix.start_index)
    {
        int j_local = jth_row + convolutionMatrix.start_index;
        int i_local;

        int i_max = ith_col + convolutionMatrix.start_index + convolutionMatrix.size;
        int j_max = j_local + convolutionMatrix.size;
        int rgb[3] = {0, 0, 0};
        for( int j_inc = 0; j_local < j_max; j_local++){

            i_local = i_max - convolutionMatrix.size;
            long index =  3 * (j_local * nb_cols + i_local);
            for(  ; i_local < i_max; i_local++, j_inc++ ){
                rgb[0] += convolutionMatrix.matrix[j_inc] * input[ index ];
                rgb[1] += convolutionMatrix.matrix[j_inc] * input[ index + 1 ];
                rgb[2] += convolutionMatrix.matrix[j_inc] * input[ index + 2 ];

                index += 3;
            }
        }
        for( int i = 0, j = 3 * (jth_row * nb_cols + ith_col); i < 3; i++, j++)
            output[j] = rgb[i] / convolutionMatrix.divisor;
    }

}
void ConvolutionTransform::saveOutput2Input() {
    for(int i = 0; i < 3 * nb_rows * nb_cols; i++) input[i] = output[i];
}
int ConvolutionTransform::getNbRows() const { return nb_rows; }
int ConvolutionTransform::getNbcols() const { return nb_cols; }

ConvolutionTransform::~ConvolutionTransform() = default;





