#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

#include "common/menu_lib.hpp"
#include "common/utilities.hpp"
#include "gpu/gpu_img_transform.cuh"

int main(int argc, char **argv)
{
    std::string img_out = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/out.jpg";
    std::string img_in = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/in.jpg";

    MenuSelection menuSelection;
    initParameters(img_in, img_out, menuSelection, argc, argv);
    ExecutionInfo info;

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);

    //std::vector< unsigned char > g( rows * cols );
    // Allocation de l'image de sortie en RAM côté CPU.
    auto rows = m_in.rows;
    auto cols = m_in.cols;
    unsigned char* rgb_in_aux = nullptr;
    cudaMallocHost(&rgb_in_aux, 3 * m_in.rows * m_in.cols);
    cv::Mat m_out(m_in.rows, m_in.cols, CV_8UC3, rgb_in_aux);

    if( ! menuSelection.use_shared ){
        for( int i = 0; i < menuSelection.enabled_filters.size(); i++){
            EffectStyle filter = menuSelection.enabled_filters.at(i);
            set_convolution_properties(info.conv_properties, filter);
            int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

            char conv_mat[conv_mat_length];
            info.conv_matrix = conv_mat;
            info.nb_pass = menuSelection.nb_pass.at(i);
            info.block.x = menuSelection.block.dimX;
            info.block.y = menuSelection.block.dimY;

            copyReverse(conv_mat, filter, conv_mat_length);

            GpuImgTransform::execute(m_in, m_out, info );

            memcpy(m_in.data, m_out.data, 3 * rows * cols  * sizeof(unsigned char));

        }
    }else{
        for( int i = 0; i < menuSelection.enabled_filters.size(); i++){
            EffectStyle filter = menuSelection.enabled_filters.at(i);
            set_convolution_properties(info.conv_properties, filter);
            int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

            char conv_mat[conv_mat_length];
            info.conv_matrix = conv_mat;
            info.nb_pass = menuSelection.nb_pass.at(i);
            info.block.x = menuSelection.block.dimX;
            info.block.y = menuSelection.block.dimY;

            copyReverse(conv_mat, filter, conv_mat_length);

            GpuImgTransform::executeSharedMemMode(m_in, m_out, info);

            memcpy(m_in.data, m_out.data, 3 * rows * cols  * sizeof(unsigned char));

        }

    }
    cv::imwrite(img_out, m_out);

    cudaFreeHost(rgb_in_aux);

    return 0;
}
