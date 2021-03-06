#include <cstdio>
#include "convolution_transform.hpp"

void ConvolutionMatrix::transformPixel(int ith_col, int jth_row) {
    if (ith_col > 0 && ith_col < nb_cols + start_index && jth_row > 0 && jth_row < nb_rows + start_index)
    {
        int j_local = jth_row + start_index;
        int i_local;

        int i_max = ith_col + start_index + size;
        int j_max = j_local + size;
        int rgb[3] = {0, 0, 0};
        for( int j_inc = 0; j_local < j_max; j_local++){

            i_local = i_max - size;
            long index =  3 * (j_local * nb_cols + i_local);
            for(  ; i_local < i_max; i_local++, j_inc++ ){
                rgb[0] += mat[j_inc] * input[ index ];
                rgb[1] += mat[j_inc] * input[ index + 1 ];
                rgb[2] += mat[j_inc] * input[ index + 2 ];

                index += 3;
            }
        }
        for( int i = 0, j = 3 * (jth_row * nb_cols + ith_col); i < 3; i++, j++) output[j] = rgb[i] / divisor;
    }

}

ConvolutionTransform::ConvolutionTransform( unsigned char *input, int nb_cols, int nb_rows, EffectStyle style) {
    switch( style){
        case EMBOSS5:
            convolutionMatrix = new ConvolutionMatrix(5, input, nb_cols, nb_rows);
            convolutionMatrix->setConvMatrix(
                    new char[5*5]{
                                 1, 0, 0, 0, 0,
                                 0, 1, 0, 0, 0,
                                 0, 0, 0, 0, 0,
                                 0, 0, 0, -1, 0,
                                 0, 0, 0, 0, -1
                    }, 1);
            break;
        case EMBOSS:
            convolutionMatrix = new ConvolutionMatrix(3, input, nb_cols, nb_rows);
            convolutionMatrix->setConvMatrix(
                    new char[3*3]{
                            -2, -1, 0,
                            -1, 1, 1,
                            0, 1, 2
                    }, 1);
            break;
        case GAUSSIANBLUR5:
            convolutionMatrix = new ConvolutionMatrix(5, input, nb_cols, nb_rows);
            convolutionMatrix->setConvMatrix(
                    new char[5*5]{
                                 1, 4, 6, 4, 1,
                                 4, 16, 24, 16, 4,
                                 6, 24, 36, 24, 6,
                                 4, 16, 24, 16, 4,
                                 1, 4, 6, 4, 1
                    }, 256);
            break;
        case GAUSSIANBLUR:
            convolutionMatrix = new ConvolutionMatrix(3, input, nb_cols, nb_rows);
            convolutionMatrix->setConvMatrix(
                    new char[3*3]{
                            1, 2, 1,
                            2, 4, 2,
                            1, 2, 1
                    }, 16);
            break;
        case SHARPEN:
            convolutionMatrix = new ConvolutionMatrix(3, input, nb_cols, nb_rows);
            convolutionMatrix->setConvMatrix(
                    new char[3*3]{
                            0, -5, 0,
                            -5, 3, -5,
                            0, -5, 0
                    }, -17);
            break;
        default:
            convolutionMatrix = new ConvolutionMatrix(3, input, nb_cols, nb_rows);
            convolutionMatrix->setConvMatrix(
                    new char[9]{
                            1, 1, 1,
                            1, 1, 1,
                            1, 1, 1
                    }, 9);
    }
}
void ConvolutionMatrix::saveOutput2Input() {
    for(int i = 0; i < 3 * nb_rows * nb_cols; i++) input[i] = output[i];
}
int ConvolutionMatrix::getNbRows() const { return nb_rows; }
int ConvolutionMatrix::getNbcols() const { return nb_cols; }

void ConvolutionMatrix::setConvMatrix(char *mat_arg, int div_val) {
    int size_max = size * size;
    for( int i = 0, j = size_max-1; i < size_max; i++, j--){
        mat[i] = mat_arg[j];
    }
    this->divisor = div_val;


}

ConvolutionMatrix::~ConvolutionMatrix() = default;

void ConvolutionTransform::transform() {
    int nb_rows = convolutionMatrix->getNbRows();
    int nb_cols = convolutionMatrix->getNbcols();

    for(int j = 0; j < nb_rows; j++){
        for(int i = 0; i < nb_cols; i++)
            convolutionMatrix->transformPixel(i, j);
    }
}

unsigned char *ConvolutionTransform::getResult() {
    return convolutionMatrix->output;
}

void ConvolutionTransform::transform(int nb_pass) {

    for( int i = 0; i < nb_pass; i++){
        transform();
        convolutionMatrix->saveOutput2Input();
    }

}



