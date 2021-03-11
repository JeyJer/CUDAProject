#include "gpu/gpu_img_transform_stream.cuh"

StreamInfo::StreamInfo(double size, long rows): size(size), rows(rows){
    size_bytes = (long)size * (long)sizeof(unsigned char);
    size_block = size;
}
StreamInfo::StreamInfo(const StreamInfo &streams_info, int nb_streams, long cols, ConvolutionMatrixProperties &conv_prop){
    if( nb_streams == 1){
        size_bytes = streams_info.size_bytes;
        size_block = streams_info.size;
        size = streams_info.size;
        rows = streams_info.rows;
        size_block = size;
    } else {
        // add k rows per stream
        rows = streams_info.rows / nb_streams - 2 * conv_prop.start_index;

        size_block = streams_info.size / nb_streams;

        size = streams_info.size / nb_streams - 2 * (double)conv_prop.start_index * (double)cols * 3;
        size_bytes = (long)size * (long)sizeof(unsigned char);
    }
}

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
        long j_desti = jth_row + conv_mat_properties->start_index;

        for( int i = 0, j = 3 * (j_desti * (long)nb_cols + ith_col); i < 3; i++, j++)
            output[j] = rgb[i] / conv_mat_properties->divisor;
    }
}

void GpuImgTransformStream::initMemory(cv::Mat &m_in, ConvMatrixPointers &dev_convolution,
        Pointers &host, long size, int conv_mat_length){
    cudaMallocHost(&host.rgb.in, size);
    std::memcpy(host.rgb.in, m_in.data, size);

    cudaMalloc(&dev_convolution.matrix, conv_mat_length * sizeof(char));
    cudaMalloc(&dev_convolution.prop, sizeof(ConvolutionMatrixProperties));

    cudaMemcpy(dev_convolution.matrix, host.convolution.matrix, conv_mat_length, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_convolution.prop , host.convolution.prop, sizeof(ConvolutionMatrixProperties),
            cudaMemcpyHostToDevice);

}

void GpuImgTransformStream::initStreamAndDevMem(StreamInfo &per_stream_info, int nb_streams, cudaStream_t *streams,
        RgbPointers *dev_rgbs, unsigned char *host_rgb_in){

    for( int i = 0 ; i < nb_streams; ++i ) {
        cudaMalloc(&dev_rgbs[i].in, per_stream_info.size_bytes);
        cudaMalloc(&dev_rgbs[i].out, per_stream_info.size_bytes);
    }

    for( int i = 0 ; i < nb_streams; ++i ){
        cudaMemcpyAsync( dev_rgbs[i].in,host_rgb_in + (int)(i * per_stream_info.size_block),
                         per_stream_info.size_bytes,
                         cudaMemcpyHostToDevice, streams[ i ] );
    }
}
void GpuImgTransformStream::swapStreamMem(StreamInfo &per_stream_info, int nb_streams, RgbPointers *dev_rgbs){
    for( int i = 0 ; i < nb_streams; ++i )  swapPointers(&dev_rgbs[i].in, &dev_rgbs[i].out);

    long size_copy_bytes = (long)(per_stream_info.size - per_stream_info.size_block) * (long)sizeof(unsigned char);
    for(int i = 0; i < nb_streams - 1; ++i)
        cudaMemcpyAsync(dev_rgbs[i].in + (long)per_stream_info.size_block, dev_rgbs[i+1].in,
                        size_copy_bytes, cudaMemcpyDeviceToDevice);

}
void GpuImgTransformStream::freeMemory(RgbPointers *dev_rgbs, ConvMatrixPointers &dev_convolution, Pointers &host, int nb_streams){
    for( int i = 0 ; i < nb_streams; ++i ){
        cudaFree(dev_rgbs[i].in);
        cudaFree(dev_rgbs[i].in);
    }
    cudaFree(dev_convolution.matrix);
    cudaFree(dev_convolution.prop);

    cudaFreeHost(host.rgb.in);
}

//static int execute(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
// static int executeSharedMemMode(cv::Mat &img_in, cv::Mat &img_out, char *conv_mat, ConvolutionMatrixProperties &conv_mat_prop);
int GpuImgTransformStream::execute(cv::Mat &m_in, cv::Mat &m_out, GpuUtilExecutionInfo &info)
{
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    // Pointers dev;
    Pointers host;
    ConvMatrixPointers dev_convolution;
    RgbPointers dev_rgbs[ info.nb_streams ];

    cudaStream_t streams[ info.nb_streams  ];

    host.convolution.prop = &info.conv_properties;
    host.convolution.matrix = info.conv_matrix;

    long i = 3 * rows * cols;
    StreamInfo streams_info((double)i,  m_in.rows);

    StreamInfo per_stream_info(streams_info, info.nb_streams, cols, info.conv_properties);

    int conv_mat_length = info.conv_properties.size * info.conv_properties.size;

    host.convolution.prop = &info.conv_properties;
    host.convolution.matrix = info.conv_matrix;

    GpuImgTransformStream::initMemory(m_in, dev_convolution, host, i,  conv_mat_length);


    for( i = 0 ; i < info.nb_streams ; ++i ) cudaStreamCreate( &streams[ i ] );

    GpuImgTransformStream::initStreamAndDevMem(per_stream_info, info.nb_streams, streams, dev_rgbs, host.rgb.in);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    dim3 grid0((cols - 1) / (info.block.x - 1 + info.conv_properties.start_index) + 1,
               (rows - 1) / (info.block.y - 1 + info.conv_properties.start_index) + 1);


    for( i = 0 ; i < info.nb_streams ; ++i )
    {
        transform_img_stream<<< grid0, info.block, 3 * info.block.x * info.block.y, streams[ i ] >>>(
                dev_rgbs[i].in, dev_rgbs[i].out,
                cols, per_stream_info.rows , dev_convolution.matrix, dev_convolution.prop);
    }


    for( int kth_pass = 1; kth_pass < info.nb_pass; kth_pass++){
        GpuImgTransformStream::swapStreamMem(per_stream_info, info.nb_streams, dev_rgbs);
        for( i = 0 ; i < info.nb_streams ; ++i )
        {
            transform_img_stream<<< grid0, info.block, 3 * info.block.x * info.block.y, streams[ i ] >>>(
                    dev_rgbs[i].in, dev_rgbs[i].out,
                    cols, per_stream_info.rows , dev_convolution.matrix, dev_convolution.prop);
        }
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    for(  i = 0 ; i < info.nb_streams ; ++i )
    {
        unsigned char *desti = m_out.data + (long)(i * (long)per_stream_info.size_block);

        cudaMemcpyAsync( desti, dev_rgbs[i].out,(long)per_stream_info.size_block * sizeof(unsigned char),
                cudaMemcpyDeviceToHost, streams[ i ] );
    }

    cudaEventSynchronize(stop);

    for( i = 0 ; i < info.nb_streams ; ++i ) cudaStreamDestroy( streams[ i ] );

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    GpuImgTransformStream::freeMemory(dev_rgbs, dev_convolution, host, info.nb_streams);
    return 0;
}

// TODO: stream mode shared
int GpuImgTransformStream::executeSharedMemMode(cv::Mat &m_in, cv::Mat &m_out, GpuUtilExecutionInfo &info)
{
    return 0;
}

