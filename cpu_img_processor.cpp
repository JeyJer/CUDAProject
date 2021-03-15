#include <opencv2/opencv.hpp>
#include "cpu/cpu_img_transform.hpp"

using namespace std;

int main(int argc, char **argv)
{
    string img_out;  // default values in utilities.hpp
    string img_in;

    CpuUtilMenuSelection menuSelection;
    CpuUtilMenuSelection::initParameters(img_in, img_out, menuSelection, argc, argv);

    CpuUtilExecutionInfo info;

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    unsigned char *rgb_result;

    rgb_result = (unsigned char *)malloc(3 * rows * cols * sizeof(unsigned char));

    memcpy(rgb_result, m_in.data, 3 * rows * cols  * sizeof(unsigned char));

    cv::Mat m_out(rows, cols, CV_8UC3, rgb_result);

    for( int i = 0; i < menuSelection.enabled_filters.size(); i++){
        EffectStyle filter = menuSelection.enabled_filters.at(i);
        set_convolution_properties(info.conv_properties, filter);
        int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

        char conv_mat[conv_mat_length];

        info.conv_matrix = conv_mat;
        info.nb_pass = menuSelection.nb_pass.at(i);

        // info.nb_pass = 20;
        copyReverse(conv_mat, filter, conv_mat_length);

        CpuImgTransform::execute(m_in, m_out, info );
    }

    cv::imwrite(img_out, m_out);

    free( rgb_result);

    return 0;
}
