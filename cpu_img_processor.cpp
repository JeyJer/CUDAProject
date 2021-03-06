#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include "convolution_transform.hpp"

using namespace std;
int main()
{
    auto img_out = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/out.jpg";
    auto img_in = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/in.jpg";

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    ConvolutionTransform convolutionTransform(m_in.data,  m_in.cols, m_in.rows,  SHARPEN);
    convolutionTransform.transform();
    unsigned char *g = convolutionTransform.getResult();

    cv::Mat m_out(rows, cols, CV_8UC3, g);

    cv::imwrite(img_out, m_out);


    return 0;
}
