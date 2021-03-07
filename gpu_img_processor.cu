#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

#include "common/convmatrix_lib.hpp"
#include "gpu/gpu_img_transform_v0.cuh"

int main()
{
    auto img_out = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/out.jpg";
    auto img_in = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/in.jpg";

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);

    //std::vector< unsigned char > g( rows * cols );
    // Allocation de l'image de sortie en RAM côté CPU.
    unsigned char* g = nullptr;
    cudaMallocHost(&g, 3 * m_in.rows * m_in.cols);
    cv::Mat m_out(m_in.rows, m_in.cols, CV_8UC3, g);

    GpuImgTransformV0::execute(m_in, m_out);
    cv::imwrite(img_out, m_out);

    cudaFreeHost(g);

    return 0;
}
