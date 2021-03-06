#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include "convolution_transform.hpp"

using namespace std;
class CpuImgProcessor {
    ConvolutionMatrix convolutionMatrix;
    ConvolutionTransform transformation;
public:
    CpuImgProcessor(unsigned char *input, int nb_cols, int nb_rows, EffectStyle style): convolutionMatrix(style),
            transformation(convolutionMatrix, input, nb_cols, nb_rows){
    }
    void transformImg() {
        int nb_rows = transformation.getNbRows();
        int nb_cols = transformation.getNbcols();

        for(int j = 0; j < nb_rows; j++){
            for(int i = 0; i < nb_cols; i++)
                transformation.transformPixel(i, j);
        }
    }

    unsigned char *getImgResult() {
        return transformation.output;
    }

    void transformImg(int nb_pass) {

        for( int i = 0; i < nb_pass; i++){
            transformImg();
            transformation.saveOutput2Input();
        }
    }
};


int main()
{
    auto img_out = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/out.jpg";
    auto img_in = "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/in2.jpg";

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    CpuImgProcessor cpuImgProcessor(m_in.data,  m_in.cols, m_in.rows,  GAUSSIANBLUR);
    cpuImgProcessor.transformImg(20);
    unsigned char *g = cpuImgProcessor.getImgResult();

    cv::Mat m_out(rows, cols, CV_8UC3, g);

    cv::imwrite(img_out, m_out);


    return 0;
}
