#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include "common/utilities.hpp"
#include "common/menu_lib.hpp"

using namespace std;
class CpuConvolutionTransform {
public:
    CpuConvolutionTransform(char *conv_mat, ConvolutionMatrixProperties &conv_prop,
                            unsigned char *input, int nb_cols, int nb_rows):
            input(input), nb_rows(nb_rows), nb_cols(nb_cols), output(new unsigned char[nb_cols * nb_rows * 3]),
            conv_mat(conv_mat), conv_prop(conv_prop){}

    ~CpuConvolutionTransform() = default;

    unsigned char *output;
private:
    char *conv_mat;
    unsigned char *input;
    int nb_cols;
    int nb_rows;
    ConvolutionMatrixProperties conv_prop;
public:
    void transformPixel( int ith_col, int jth_row) {
        if (ith_col + conv_prop.start_index  >= 0 && ith_col < nb_cols + conv_prop.start_index &&
            jth_row + conv_prop.start_index >= 0 && jth_row < nb_rows + conv_prop.start_index)
        {
            long j_local = jth_row + conv_prop.start_index;
            long i_local;

            long i_max = ith_col + conv_prop.start_index + conv_prop.size;
            long j_max = j_local + conv_prop.size;
            long rgb[3] = {0, 0, 0};
            for( long j_inc = 0; j_local < j_max; j_local++){

                i_local = i_max - conv_prop.size;
                long index =  3 * (j_local * nb_cols + i_local);
                for(  ; i_local < i_max; i_local++, j_inc++ ){
                    rgb[0] += conv_mat[j_inc] * input[ index ];
                    rgb[1] += conv_mat[j_inc] * input[ index + 1 ];
                    rgb[2] += conv_mat[j_inc] * input[ index + 2 ];

                    index += 3;
                }
            }
            for( long i = 0, j = 3 * (jth_row * nb_cols + ith_col); i < 3; i++, j++)
                output[j] = rgb[i] / conv_prop.divisor;
        }

    }
    void saveOutput2Input(){
        for(int i = 0; i < 3 * nb_rows * nb_cols; i++) input[i] = output[i];
    }
};


int main(int argc, char **argv)
{
    string img_out = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/out.jpg";
    string img_in = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/in.jpg";

    MenuSelection menuSelection;
    initParameters(img_in, img_out, menuSelection, argc, argv);

    ConvolutionMatrixProperties conv_properties;

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    unsigned char *rgb_in_aux;

    rgb_in_aux = (unsigned char *)malloc(3 * rows * cols * sizeof(unsigned char));

    memcpy(rgb_in_aux, m_in.data, 3 * rows * cols  * sizeof(unsigned char));

    for( int i = 0; i < menuSelection.enabled_filters.size(); i++){
        EffectStyle filter = menuSelection.enabled_filters.at(i);
        set_convolution_properties(conv_properties, filter);
        int conv_mat_length = conv_properties.size * conv_properties.size;
        char conv_mat[conv_mat_length];
        copyReverse(conv_mat, filter, conv_mat_length);

        CpuConvolutionTransform transformation(conv_mat, conv_properties, rgb_in_aux, m_in.cols, m_in.rows);

        for( int kth_pass = 0; kth_pass < menuSelection.nb_pass.at(i); kth_pass++){
            for(int  jth_row = 0; jth_row < rows; jth_row++){
                for(int ith_col = 0; ith_col < cols; ith_col++)
                    transformation.transformPixel(ith_col, jth_row);
            }
            transformation.saveOutput2Input();
        }

        memcpy(rgb_in_aux, transformation.output, 3 * rows * cols  * sizeof(unsigned char));

    }
    // TODO: free

    cv::Mat m_out(rows, cols, CV_8UC3, rgb_in_aux);
    cv::imwrite(img_out, m_out);


    return 0;
}
