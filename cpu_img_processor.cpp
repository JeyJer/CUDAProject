#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include "cpu/cpu_convolution_transform.hpp"

using namespace std;

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
        set_convolution_properties(conv_properties, menuSelection.enabled_filters.at(i));
        int conv_mat_length = conv_properties.size * conv_properties.size;
        char conv_mat[conv_mat_length];
        copyReverse(conv_mat, ARR_BOXBLUR, conv_mat_length);

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

    cv::Mat m_out(rows, cols, CV_8UC3, rgb_in_aux);
    cv::imwrite(img_out, m_out);


    return 0;
}
