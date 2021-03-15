#include "gpu_img_transform.cuh"

__global__ void transform_img(unsigned char* input, unsigned char* output, std::size_t nb_cols, std::size_t nb_rows,
                              char * conv_mat, ConvolutionMatrixProperties *conv_mat_properties)
{
    long ith_col = blockIdx.x * blockDim.x + threadIdx.x;
    long jth_row = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void transform_img_shared(unsigned char* input, unsigned char* output,
                                     std::size_t nb_cols_global, std::size_t nb_rows_global,
                                     char * conv_mat, ConvolutionMatrixProperties *conv_prop)
{
    extern __shared__ unsigned char sh[];

    long ith_col_global = blockIdx.x * (blockDim.x + conv_prop->start_index - 1) + threadIdx.x;
    long jth_row_global = blockIdx.y * (blockDim.y + conv_prop->start_index - 1) + threadIdx.y;

    long ith_col = threadIdx.x;
    long jth_row = threadIdx.y;

    long nb_rows = blockDim.y;
    long nb_cols = blockDim.x;

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
        for( long i = 0, j = 3 * (jth_row_global * (long)nb_cols_global + ith_col_global); i < 3; i++, j++)
            output[j] = rgb[i] / conv_prop->divisor;
    }
}

void GpuImgTransform::initMemory(cv::Mat &m_in, Pointers &dev, Pointers &host, long size, int conv_mat_length){
    cudaMallocHost(&host.rgb.in, size);
    std::memcpy(host.rgb.in, m_in.data, size);

    cudaMalloc(&dev.rgb.in, size);
    cudaMalloc(&dev.rgb.out, size);
    cudaMalloc(&dev.convolution.matrix, conv_mat_length * sizeof(char));
    cudaMalloc(&dev.convolution.prop, sizeof(ConvolutionMatrixProperties));

    cudaMemcpy(dev.rgb.in, host.rgb.in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev.convolution.matrix, host.convolution.matrix, conv_mat_length, cudaMemcpyHostToDevice);
    cudaMemcpy(dev.convolution.prop , host.convolution.prop, sizeof(ConvolutionMatrixProperties), cudaMemcpyHostToDevice);
}
void GpuImgTransform::freeMemory(Pointers &dev, Pointers &host){
    cudaFree(dev.rgb.in);
    cudaFree(dev.rgb.out);
    cudaFree(dev.convolution.matrix);
    cudaFree(dev.convolution.prop);

    cudaFreeHost(host.rgb.in);
}

int GpuImgTransform::execute(cv::Mat &m_in, cv::Mat &m_out, GpuUtilExecutionInfo &info)
{
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    Pointers dev;
    Pointers host;

    host.convolution.prop = &info.conv_properties;
    host.convolution.matrix = info.conv_matrix;

    int size = 3 * rows * cols;
    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    initMemory(m_in, dev, host, size, conv_mat_length);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    dim3 grid0((cols - 1) / info.block.x + 1, (rows - 1) / info.block.y + 1);

    transform_img<<< grid0, info.block >>>(dev.rgb.in, dev.rgb.out, cols, rows, dev.convolution.matrix,
            dev.convolution.prop);

    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++){
        swapPointers(&dev.rgb.in, &dev.rgb.out);
        transform_img<<< grid0, info.block >>>(dev.rgb.in, dev.rgb.out, cols, rows, dev.convolution.matrix,
                                               dev.convolution.prop);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(m_out.data, dev.rgb.out, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    freeMemory(dev, host);
    return 0;
}

int GpuImgTransform::executeSharedMemMode(cv::Mat &m_in, cv::Mat &m_out, GpuUtilExecutionInfo &info){
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    Pointers dev;
    Pointers host;

    host.convolution.prop = &info.conv_properties;
    host.convolution.matrix = info.conv_matrix;

    int size = 3 * rows * cols;
    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    initMemory(m_in, dev, host, size, conv_mat_length);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    dim3 grid0((cols - 1) / (info.block.x - 1 + info.conv_properties.start_index) + 1,
            (rows - 1) / (info.block.y - 1 + info.conv_properties.start_index) + 1);

    transform_img_shared<<<grid0, info.block, 3 * info.block.x * info.block.y>>>(dev.rgb.in, dev.rgb.out, cols, rows,
            dev.convolution.matrix, dev.convolution.prop);

    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++){
        swapPointers(&dev.rgb.in, &dev.rgb.out);
        transform_img_shared<<<grid0, info.block, 3 * info.block.x * info.block.y>>>(dev.rgb.in, dev.rgb.out, cols, rows,
                dev.convolution.matrix, dev.convolution.prop);

    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaMemcpy(m_out.data, dev.rgb.out, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    freeMemory(dev, host);
    return 0;
}
