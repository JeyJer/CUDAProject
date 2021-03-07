#include "gpu/gpu_img_transform.cuh"
#include "gpu/gpu_img_transform_stream.cuh"

__global__ void transform_img_stream(unsigned char* input, unsigned char* output, std::size_t nb_cols, std::size_t nb_rows,
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
        for( int j_inc = 0; j_local < j_max; j_local++){

            i_local = i_max - conv_mat_properties->size;
            long index =  3 * (j_local * (long)nb_cols + i_local);
            for(  ; i_local < i_max; i_local++, j_inc++ ){
                rgb[0] += conv_mat[j_inc] * input[ index ];
                rgb[1] += conv_mat[j_inc] * input[ index + 1 ];
                rgb[2] += conv_mat[j_inc] * input[ index + 2 ];

                index += 3;
            }
        }
        for( int i = 0, j = 3 * (jth_row * (long)nb_cols + ith_col); i < 3; i++, j++)
            output[j] = rgb[i] / conv_mat_properties->divisor;
    }
}

void initSteamModeMemory(cv::Mat &m_in, RefPointers &dev, RefPointers &host, long size, int conv_mat_length){
    cudaMallocHost(&host.rgb_in, size);
    std::memcpy(host.rgb_in, m_in.data, size);

    cudaMalloc(&dev.rgb_in, size);
    cudaMalloc(&dev.rgb_out, size);
    cudaMalloc(&dev.conv_matrix, conv_mat_length * sizeof(char));
    cudaMalloc(&dev.conv_prop, sizeof(ConvolutionMatrixProperties));

    cudaMemcpy(dev.conv_matrix, host.conv_matrix, conv_mat_length, cudaMemcpyHostToDevice);
    cudaMemcpy(dev.conv_prop , host.conv_prop, sizeof(ConvolutionMatrixProperties), cudaMemcpyHostToDevice);
}

void initStreamAndDevMem(StreamProperty &size, int nb_streams, cudaStream_t *streams,
        unsigned char *dev_rgb_in, unsigned char *host_rgb_in){
    double size_per_stream = (double)size.units / nb_streams;
    long size_per_stream_bytes = size.bytes / nb_streams;
    for( int i = 0 ; i < nb_streams; ++i ){

        cudaMemcpyAsync( dev_rgb_in + (int)(i * size_per_stream) , host_rgb_in + (int)(i * size_per_stream),
                         size_per_stream_bytes, cudaMemcpyHostToDevice, streams[ i ] );
    }

}
void freeStreamModeMemory(RefPointers &dev, RefPointers &host){
    cudaFree(dev.rgb_in);
    cudaFree(dev.rgb_out);
    cudaFree(dev.conv_matrix);
    cudaFree(dev.conv_prop);

    cudaFreeHost(host.rgb_in);
}

//static int execute(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
// static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
int GpuImgTransformStream::execute(cv::Mat &m_in, cv::Mat &m_out, ExecutionInfo &info)
{
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    RefPointers dev;
    RefPointers host;

    host.conv_prop = &info.conv_properties;
    host.conv_matrix = info.conv_matrix;

    long i = 3 * rows * cols;
    StreamProperty size(i, i * (int)sizeof(unsigned char));
    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    initSteamModeMemory(m_in, dev, host, size.units, conv_mat_length);

    host.conv_prop = &info.conv_properties;
    host.conv_matrix = info.conv_matrix;

    cudaStream_t streams[ info.nb_streams  ];
    for( i = 0 ; i < info.nb_streams ; ++i ) cudaStreamCreate( &streams[ i ] );

    initStreamAndDevMem(size, info.nb_streams, streams, dev.rgb_in, host.rgb_in);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    dim3 grid0((cols - 1) / (info.block.x - 1 + info.conv_properties.start_index) + 1,
               (rows - 1) / (info.block.y - 1 + + info.conv_properties.start_index) + 1);

    double size_per_stream = (double)size.units / info.nb_streams;
    long size_per_stream_bytes = size.bytes / info.nb_streams;
    long rows_per_stream = rows / info.nb_streams;

    for( i = 0 ; i < info.nb_streams ; ++i )
    {
        transform_img_stream<<< grid0, info.block, 3 * info.block.x * info.block.y, streams[ i ] >>>(
                dev.rgb_in + (long)(i * size_per_stream), dev.rgb_out + (long)(i * size_per_stream),
                cols, rows_per_stream , dev.conv_matrix, dev.conv_prop);
    }
    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++ ){
        swapPointers(&dev.rgb_in, &dev.rgb_out);
        for( i = 0 ; i < info.nb_streams ; ++i )
        {
            transform_img_stream<<< grid0, info.block, 3 * info.block.x * info.block.y, streams[ i ] >>>(
                    dev.rgb_in + (long)(i * size_per_stream), dev.rgb_out + (long)(i * size_per_stream),
                    cols, rows_per_stream , dev.conv_matrix, dev.conv_prop);
        }
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    // unsigned char *host_rgb_aux = m_out.data;
    for(  i = 0 ; i < info.nb_streams ; ++i )
    {
        cudaMemcpyAsync( m_out.data + (long)(i * (size_per_stream - 6 * cols)),
                dev.rgb_out + (long)(i * (size_per_stream) + 3 * cols),
                size_per_stream_bytes - 6 * cols * sizeof(unsigned char),
                cudaMemcpyDeviceToHost, streams[ i ] );

    }

    cudaEventSynchronize(stop);

    for( i = 0 ; i < info.nb_streams ; ++i ) cudaStreamDestroy( streams[ i ] );

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    freeStreamModeMemory(dev, host);
    return 0;
}

int GpuImgTransformStream::executeSharedMemMode(cv::Mat &m_in, cv::Mat &m_out, ExecutionInfo &info)
{
    return 0;
}

