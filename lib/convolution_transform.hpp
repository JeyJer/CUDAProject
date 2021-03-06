#ifndef PROJET_CUDA_CONVOLUTION_TRANSFORM_HPP
#define PROJET_CUDA_CONVOLUTION_TRANSFORM_HPP

using namespace std;

enum EffectStyle {
    BOXBLUR, GAUSSIANBLUR, GAUSSIANBLUR5, EMBOSS, EMBOSS5, SHARPEN
};
struct ConvolutionMatrix {
    char *matrix;
    int size;
    int divisor;
    int start_index;
    explicit ConvolutionMatrix(EffectStyle style);
    void set(char *mat, int size_val, int div_val);
};

class ConvolutionTransform {
public:
    ConvolutionTransform(ConvolutionMatrix &convMatrix, unsigned char *input, int nb_cols, int nb_rows);
    virtual ~ConvolutionTransform();

    unsigned char *output;

private:
    ConvolutionMatrix convolutionMatrix;
    unsigned char *input;

    int nb_cols;
    int nb_rows;
public:
    void transformPixel(int nb_cols, int nb_rows);
    void saveOutput2Input();
    int getNbcols() const;
    int getNbRows() const;
};

#endif //PROJET_CUDA_CONVOLUTION_TRANSFORM_HPP
