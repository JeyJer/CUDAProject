#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>
#include "gpu_img_transform.cuh"

void ExecutionInfo::set(char *conv_mat, int nunber_of_pass, int dimX, int dimY, int number_of_streams){
    conv_matrix = conv_mat;
    nb_pass = nunber_of_pass;
    nb_streams = number_of_streams;
    block.x = dimX;
    block.y = dimY;
}

__global__ void transform_img(unsigned char* input, unsigned char* output, std::size_t nb_cols, std::size_t nb_rows,
                              char * conv_mat, ConvolutionMatrixProperties *conv_mat_properties)
{
    long ith_col = blockIdx.x * blockDim.x + threadIdx.x;
    long jth_row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: supprimer debug msg
    if( ith_col == 0 && jth_row == 0){
        printf("size: %d; start_index: %d; divisor: %d\n;", conv_mat_properties->size, conv_mat_properties->start_index,
               conv_mat_properties->divisor);

        printf("{ %d", conv_mat[0]);
        for( int i = 1; i < conv_mat_properties->size * conv_mat_properties->size; i++){
            printf(", %d", conv_mat[i]);
        }
        printf(" }\n");
    }

    if (ith_col + conv_mat_properties->start_index >= 0 && ith_col < nb_cols + conv_mat_properties->start_index &&
        jth_row + conv_mat_properties->start_index >= 0 && jth_row < nb_rows + conv_mat_properties->start_index)
    {
        long j_local = jth_row + conv_mat_properties->start_index;
        long i_local;

        long i_max = ith_col + conv_mat_properties->start_index + conv_mat_properties->size;
        long j_max = j_local + conv_mat_properties->size;
        long rgb[3] = {0, 0, 0};
        for( long j_inc = 0; j_local < j_max; j_local++){

            i_local = i_max - conv_mat_properties->size;
            long index =  3 * (j_local * (long)nb_cols + i_local);
            for(  ; i_local < i_max; i_local++, j_inc++ ){
                rgb[0] += conv_mat[j_inc] * input[ index ];
                rgb[1] += conv_mat[j_inc] * input[ index + 1 ];
                rgb[2] += conv_mat[j_inc] * input[ index + 2 ];

                index += 3;
            }
        }
        for( long i = 0, j = 3 * (jth_row * (long)nb_cols + ith_col); i < 3; i++, j++)
            output[j] = rgb[i] / conv_mat_properties->divisor;
    }
}

// TODO: a tester
__global__ void transform_img_shared(unsigned char* input, unsigned char* output,
                                     std::size_t nb_cols_global, std::size_t nb_rows_global,
                                     char * conv_mat, ConvolutionMatrixProperties *conv_prop)
{
    long ith_col_global = blockIdx.x * blockDim.x + threadIdx.x;
    long jth_row_global = blockIdx.y * blockDim.y + threadIdx.y;

    long ith_col = threadIdx.x;
    long jth_row = threadIdx.y;

    long nb_rows = blockIdx.x;
    long nb_cols = blockIdx.y;

    // TODO: supprimer debug msg
    if( ith_col == 0 && jth_row == 0){
        printf("size: %d; start_index: %d; divisor: %d\n;", conv_prop->size, conv_prop->start_index,
               conv_prop->divisor);

        printf("{ %d", conv_mat[0]);
        for( int i = 1; i < conv_prop->size * conv_prop->size; i++){
            printf(", %d", conv_mat[i]);
        }
        printf(" }\n");
    }

    extern __shared__ unsigned char sh[];

    if (ith_col_global < nb_cols_global && jth_row_global < nb_rows_global)
    {
        long index = 3 * (jth_row * nb_cols + ith_col);
        long index_global = 3 * (jth_row_global * (long)nb_cols_global + (long)ith_col_global);
        sh[index] = input[index_global ];
        sh[index + 1] = input[index_global + 1];
        sh[index + 2] = input[index_global + 2];
    }

    __syncthreads();

    if ( ( ith_col_global > 0 && ith_col_global < nb_cols_global ) &&
         ( jth_row_global > 0 && jth_row_global < nb_rows_global) &&
         ( ith_col + conv_prop->start_index >= 0 && ith_col < nb_cols + conv_prop->start_index ) &&
         ( jth_row + conv_prop->start_index >= 0 && jth_row < nb_rows + conv_prop->start_index) )
    {
        long j_local = jth_row + conv_prop->start_index;
        long i_local;

        long i_max = ith_col + conv_prop->start_index + conv_prop->size;
        long j_max = j_local + conv_prop->size;
        long rgb[3] = {0, 0, 0};
        for( long j_inc = 0; j_local < j_max; j_local++){

            i_local = i_max - conv_prop->size;
            long index =  3 * (j_local * nb_cols + i_local);
            for(  ; i_local < i_max; i_local++, j_inc++ ){
                rgb[0] += conv_mat[j_inc] * sh[ index ];
                rgb[1] += conv_mat[j_inc] * sh[ index + 1 ];
                rgb[2] += conv_mat[j_inc] * sh[ index + 2 ];

                index += 3;
            }
        }
        for( long i = 0, j = 3 * (jth_row_global * nb_cols_global + ith_col_global); i < 3; i++, j++)
            output[j] = rgb[i] / conv_prop->divisor;
    }
}

void initMemory(cv::Mat &m_in, RefPointers &dev, RefPointers &host, long size, int conv_mat_length){
    cudaMallocHost(&host.rgb_in, size);
    std::memcpy(host.rgb_in, m_in.data, size);

    cudaMalloc(&dev.rgb_in, size);
    cudaMalloc(&dev.rgb_out, size);
    cudaMalloc(&dev.conv_matrix, conv_mat_length * sizeof(char));
    cudaMalloc(&dev.conv_prop, sizeof(ConvolutionMatrixProperties));

    cudaMemcpy(dev.rgb_in, host.rgb_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev.conv_matrix, host.conv_matrix, conv_mat_length, cudaMemcpyHostToDevice);
    cudaMemcpy(dev.conv_prop , host.conv_prop, sizeof(ConvolutionMatrixProperties), cudaMemcpyHostToDevice);
}
void freeMemory(RefPointers &dev, RefPointers &host){
    cudaFree(dev.rgb_in);
    cudaFree(dev.rgb_out);
    cudaFree(dev.conv_matrix);
    cudaFree(dev.conv_prop);

    cudaFreeHost(host.rgb_in);
}

//static int execute(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
// static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
int GpuImgTransform::execute(cv::Mat &m_in, cv::Mat &m_out, ExecutionInfo &info)
{
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    RefPointers dev;
    RefPointers host;

    host.conv_prop = &info.conv_properties;
    host.conv_matrix = info.conv_matrix;

    int size = 3 * rows * cols;
    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    initMemory(m_in, dev, host, size, conv_mat_length);

    host.conv_prop = &info.conv_properties;
    host.conv_matrix = info.conv_matrix;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    dim3 grid0((cols - 1) / info.block.x + 1, (rows - 1) / info.block.y + 1);

    transform_img<<< grid0, info.block >>>(dev.rgb_in, dev.rgb_out, cols, rows, dev.conv_matrix, dev.conv_prop);
    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++ ){
        swapPointers(&dev.rgb_in, &dev.rgb_out);
        transform_img<<< grid0, info.block >>>(dev.rgb_in, dev.rgb_out, cols, rows, dev.conv_matrix, dev.conv_prop);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // unsigned char *host_rgb_aux = m_out.data;
    cudaMemcpy(m_out.data, dev.rgb_out, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    freeMemory(dev, host);
    return 0;
}


int GpuImgTransform::executeSharedMemMode(cv::Mat &m_in, cv::Mat &m_out, ExecutionInfo &info){
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    RefPointers dev;
    RefPointers host;

    host.conv_prop = &info.conv_properties;
    host.conv_matrix = info.conv_matrix;

    int size = 3 * rows * cols;
    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    initMemory(m_in, dev, host, size, conv_mat_length);

    host.conv_prop = &info.conv_properties;
    host.conv_matrix = info.conv_matrix;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    dim3 grid0((cols - 1) / (info.block.x - 1 + info.conv_properties.start_index) + 1,
            (rows - 1) / (info.block.y - 1 + + info.conv_properties.start_index) + 1);

    transform_img_shared<<<grid0, info.block, 3 * info.block.x * info.block.y>>>(dev.rgb_in, dev.rgb_out, cols, rows,
                                                                                 dev.conv_matrix, dev.conv_prop);
    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++ ){
        swapPointers(&dev.rgb_in, &dev.rgb_out);
        transform_img_shared<<<grid0, info.block, 3 * info.block.x * info.block.y>>>(dev.rgb_in, dev.rgb_out, cols, rows,
                                                                      dev.conv_matrix, dev.conv_prop);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // unsigned char *host_rgb_aux = m_out.data;
    cudaMemcpy(m_out.data, dev.rgb_out, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    freeMemory(dev, host);
    return 0;
}
