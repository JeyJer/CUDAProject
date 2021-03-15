#include "utilities.hpp"
#include "cpu_img_transform.hpp"

void CpuImgTransform::initMemory(cv::Mat &m_in, Pointers &host, long size, int conv_mat_length){
    host.rgb.in = (unsigned char *)malloc(size * sizeof(unsigned char));
    host.rgb.out = (unsigned char *)malloc(size * sizeof(unsigned char));

    std::memcpy(host.rgb.in, m_in.data, size);
}
void CpuImgTransform::freeMemory(Pointers &host){
    free( host.rgb.in );
    free( host.rgb.out );
}
void CpuImgTransform::transform_img(unsigned char* input, unsigned char* output, std::size_t nb_cols,
        std::size_t nb_rows, char *conv_mat, ConvolutionMatrixProperties &conv_prop)
{
   for(long ith_col = -conv_prop.start_index; ith_col < nb_cols + conv_prop.start_index; ith_col++)
   {
       for(long jth_row = -conv_prop.start_index; jth_row < nb_rows + conv_prop.start_index; jth_row++){
           long j_local = jth_row +  conv_prop.start_index;
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

}

//static int execute(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
// static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
int CpuImgTransform::execute(cv::Mat &m_in, cv::Mat &m_out, CpuUtilExecutionInfo &info)
{
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    Pointers host;

    host.convolution.prop = &info.conv_properties;
    host.convolution.matrix = info.conv_matrix;

    long size = 3 * rows * cols;

    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    CpuImgTransform::initMemory(m_in, host, size, conv_mat_length);

    host.convolution.prop = &info.conv_properties;
    host.convolution.matrix = info.conv_matrix;

    // start timer
    auto start_timer = std::chrono::high_resolution_clock::now();

    transform_img(host.rgb.in, host.rgb.out, cols, rows , host.convolution.matrix, *host.convolution.prop);

    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++){
        swapPointers(&host.rgb.in, &host.rgb.out);
        transform_img(host.rgb.in, host.rgb.out, cols, rows , host.convolution.matrix, *host.convolution.prop);
    }

    auto stop_timer = std::chrono::high_resolution_clock::now();

    float duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_timer - start_timer).count();
    // stop timer
    std::cout << "time=" << duration << " ms" << std::endl;

    memcpy(m_out.data, host.rgb.out, size * sizeof(unsigned char));

    CpuImgTransform::freeMemory(host);
    return 0;
}

