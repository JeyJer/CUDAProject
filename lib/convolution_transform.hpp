#ifndef PROJET_CUDA_CONVOLUTION_TRANSFORM_HPP
#define PROJET_CUDA_CONVOLUTION_TRANSFORM_HPP

using namespace std;

enum EffectStyle {
    BOXBLUR, GAUSSIANBLUR, GAUSSIANBLUR5, EMBOSS, EMBOSS5, SHARPEN
};

class ConvolutionMatrix {
public:
    ConvolutionMatrix(int size, unsigned char *input, int nb_cols, int nb_rows): size(size), input(input),
        nb_rows(nb_rows), nb_cols(nb_cols), output(new unsigned char[nb_cols * nb_rows * 3]),
        start_index(-(size-1) / 2), divisor(1), mat(new char[size * size]) {}
    virtual ~ConvolutionMatrix();

    unsigned char *output;

protected:

    unsigned char *input;

    int divisor;
    int nb_cols;
    int nb_rows;
    int size;
    int start_index;
    char *mat;
public:
    void setConvMatrix(char *mat, int div_val);
    void transformPixel(int nb_cols, int nb_rows);
    void saveOutput2Input();
    int getNbcols() const;
    int getNbRows() const;
};

class ConvolutionTransform {

    ConvolutionMatrix *convolutionMatrix;
public:
    ConvolutionTransform(unsigned char *input, int nb_cols, int nb_rows, EffectStyle style);
    void transform();
    void transform(int nb_pass);
    unsigned char *getResult();

};

#endif //PROJET_CUDA_CONVOLUTION_TRANSFORM_HPP
