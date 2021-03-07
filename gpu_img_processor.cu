#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include <gpu/gpu_img_transform_stream.cuh>

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

    int (*fnc_exec) (cv::Mat&, cv::Mat&, ExecutionInfo& );

    // menuSelection.nb_stream = 20;
    if( menuSelection.nb_stream == 0) {
        if (!menuSelection.use_shared)
            fnc_exec = GpuImgTransform::execute;
        else
            fnc_exec = GpuImgTransform::executeSharedMemMode;
    }else {
        if (!menuSelection.use_shared)
            fnc_exec = GpuImgTransformStream::execute;
        else
            fnc_exec = GpuImgTransformStream::executeSharedMemMode;
    }
    for( int i = 0; i < menuSelection.enabled_filters.size(); i++){
        EffectStyle filter = menuSelection.enabled_filters.at(i);
        set_convolution_properties(info.conv_properties, filter);
        int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

        char conv_mat[conv_mat_length];
        info.set(conv_mat, menuSelection.nb_pass.at(i), menuSelection.block.dimX,
                 menuSelection.block.dimY, menuSelection.nb_stream);

        copyReverse(conv_mat, filter, conv_mat_length);

        (*fnc_exec)(m_in, m_out, info );

        memcpy(m_in.data, m_out.data, 3 * rows * cols  * sizeof(unsigned char));

    }

    cv::imwrite(img_out, m_out);

    cudaFreeHost(rgb_in_aux);

    return 0;
}
