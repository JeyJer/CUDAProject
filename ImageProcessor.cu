#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

__global__ void flou(unsigned char* rgb, unsigned char* s, std::size_t cols, std::size_t rows)
{

    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    /*
     * int matrix[3][3] = {
      { 1, 2, 1 },
      { 2, 4, 2 },
      { 1, 2, 1 }
    };
     */

    unsigned char matrix[3][3] = {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 }
    };
    int diviseur = 9;

    if (i > 0 && i < cols && j > 0 && j < rows)
    {
        auto h = matrix[0][0] * rgb[3 * ((j - 1) * cols + i - 1)] + matrix[0][1] * rgb[3 * ((j - 1) * cols + i)] + matrix[0][2] * rgb[3 * ((j - 1) * cols + i + 1)]
            + matrix[1][0] * rgb[3 * ((j)*cols + i - 1)] + matrix[1][1] * rgb[3 * ((j)*cols + i)] + matrix[1][2] * rgb[3 * ((j)*cols + i + 1)]
            + matrix[2][0] * rgb[3 * ((j + 1) * cols + i - 1)] + matrix[2][1] * rgb[3 * ((j + 1) * cols + i)] + matrix[2][2] * rgb[3 * ((j + 1) * cols + i + 1)];

        auto h_g = matrix[0][0] * rgb[3 * ((j - 1) * cols + i - 1) + 1] + matrix[0][1] * rgb[3 * ((j - 1) * cols + i) + 1] + matrix[0][2] * rgb[3 * ((j - 1) * cols + i + 1) + 1]
            + matrix[1][0] * rgb[3 * ((j)*cols + i - 1) + 1] + matrix[1][1] * rgb[3 * ((j)*cols + i) + 1] + matrix[1][2] * rgb[3 * ((j)*cols + i + 1) + 1]
            + matrix[2][0] * rgb[3 * ((j + 1) * cols + i - 1) + 1] + matrix[2][1] * rgb[3 * ((j + 1) * cols + i) + 1] + matrix[2][2] * rgb[3 * ((j + 1) * cols + i + 1) + 1];

        auto h_b = matrix[0][0] * rgb[3 * ((j - 1) * cols + i - 1) + 2] + matrix[0][1] * rgb[3 * ((j - 1) * cols + i) + 2] + matrix[0][2] * rgb[3 * ((j - 1) * cols + i + 1) + 2]
            + matrix[1][0] * rgb[3 * ((j)*cols + i - 1) + 2] + matrix[1][1] * rgb[3 * ((j)*cols + i) + 2] + matrix[1][2] * rgb[3 * ((j)*cols + i + 1) + 2]
            + matrix[2][0] * rgb[3 * ((j + 1) * cols + i - 1) + 2] + matrix[2][1] * rgb[3 * ((j + 1) * cols + i) + 2] + matrix[2][2] * rgb[3 * ((j + 1) * cols + i + 1) + 2];


        s[3 * (j * cols + i)] = (h / diviseur);
        s[3 * (j * cols + i) + 1] = (h_g / diviseur);
        s[3 * (j * cols + i) + 2] = (h_b / diviseur);

    }
}

__global__ void flou_shared(unsigned char* rgb, unsigned char* s, std::size_t cols, std::size_t rows)
{
    auto i_global = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    auto j_global = blockIdx.y * (blockDim.y - 2) + threadIdx.y;

    auto i = threadIdx.x;
    auto j = threadIdx.y;

    auto w = blockDim.x;
    auto height = blockDim.y;

    extern __shared__ unsigned char sh[];

    if (i_global < cols && j_global < rows)
    {
        sh[3 * (j * w + i)] = rgb[3 * (j_global * cols + i_global) ];
        sh[3 * (j * w + i) + 1] = rgb[3 * (j_global * cols + i_global) + 1];
        sh[3 * (j * w + i) + 2] = rgb[3* ( j_global * cols + i_global) + 2];
    }

    __syncthreads();

    /*
     * int matrix[3][3] = {
      { 1, 2, 1 },
      { 2, 4, 2 },
      { 1, 2, 1 }
    };
     */

    unsigned char matrix[3][3] = {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 }
    };
    int diviseur = 9;

    if (i_global < cols - 1 && j_global < rows - 1 && i > 0 && i < (w - 1) && j > 0 && j < (height - 1))
    {
        auto h = matrix[0][0] * sh[3 * ((j - 1) * w + i - 1)] + matrix[0][1] * sh[3 * ((j - 1) * w + i)] + matrix[0][2] * sh[3 * ((j - 1) * w + i + 1)]
                 + matrix[1][0] * sh[3 * ((j)*w + i - 1)] + matrix[1][1] * sh[3 * ((j)*w + i)] + matrix[1][2] * sh[3 * ((j)*w + i + 1)]
                 + matrix[2][0] * sh[3 * ((j + 1) * w + i - 1)] + matrix[2][1] * sh[3 * ((j + 1) * w + i)] + matrix[2][2] * sh[3 * ((j + 1) * w + i + 1)];

        auto h_g = matrix[0][0] * sh[3 * ((j - 1) * w + i - 1) + 1] + matrix[0][1] * sh[3 * ((j - 1) * w + i) + 1] + matrix[0][2] * sh[3 * ((j - 1) * w + i + 1) + 1]
                   + matrix[1][0] * sh[3 * ((j)*w + i - 1) + 1] + matrix[1][1] * sh[3 * ((j)*w + i) + 1] + matrix[1][2] * sh[3 * ((j)*w + i + 1) + 1]
                   + matrix[2][0] * sh[3 * ((j + 1) * w + i - 1) + 1] + matrix[2][1] * sh[3 * ((j + 1) * w + i) + 1] + matrix[2][2] * sh[3 * ((j + 1) * w + i + 1) + 1];

        auto h_b = matrix[0][0] * sh[3 * ((j - 1) * w + i - 1) + 2] + matrix[0][1] * sh[3 * ((j - 1) * w + i) + 2] + matrix[0][2] * sh[3 * ((j - 1) * w + i + 1) + 2]
                   + matrix[1][0] * sh[3 * ((j)*w + i - 1) + 2] + matrix[1][1] * sh[3 * ((j)*w + i) + 2] + matrix[1][2] * sh[3 * ((j)*w + i + 1) + 2]
                   + matrix[2][0] * sh[3 * ((j + 1) * w + i - 1) + 2] + matrix[2][1] * sh[3 * ((j + 1) * w + i) + 2] + matrix[2][2] * sh[3 * ((j + 1) * w + i + 1) + 2];


        s[3 * (j_global * cols + i_global)] = (h / diviseur);
        s[3 * (j_global * cols + i_global) + 1] = (h_g / diviseur);
        s[3 * (j_global * cols + i_global) + 2] = (h_b / diviseur);

    }
}

int main()
{
    auto img_out = "/mnt/data/tsky-19/eclipsec/CUDAProject_branch1/out.jpg";
    auto img_in = "/mnt/data/tsky-19/eclipsec/CUDAProject_branch1/in.jpg";

    cv::Mat m_in = cv::imread(img_in, cv::IMREAD_UNCHANGED);

    //auto rgb = m_in.data;
    auto rows = m_in.rows;
    auto cols = m_in.cols;

    //std::vector< unsigned char > g( rows * cols );
    // Allocation de l'image de sortie en RAM côté CPU.
    unsigned char* g = nullptr;
    cudaMallocHost(&g, 3 * rows * cols);
    cv::Mat m_out(rows, cols, CV_8UC3, g);

    // Copie de l'image en entrée dans une mémoire dite "pinned" de manière à accélérer les transferts.
    // OpenCV alloue la mémoire en interne lors de la décompression de l'image donc soit sans doute avec
    // un malloc standard.
    unsigned char* rgb = nullptr;
    cudaMallocHost(&rgb, 3 * rows * cols);

    std::memcpy(rgb, m_in.data, 3 * rows * cols);

    unsigned char* rgb_d;
    unsigned char* s_d;

    cudaMalloc(&rgb_d, 3 * rows * cols);
    cudaMalloc(&s_d, 3 * rows * cols);

    cudaStream_t streams[ 2 ];

    for( std::size_t i = 0 ; i < 2 ; ++i ) cudaStreamCreate( &streams[ i ] );

    int size = 3 * rows * cols;
    int size_bytes = size * (int)sizeof(unsigned char) ;

    int i = 0;
    cudaMemcpyAsync( rgb_d, rgb, size_bytes/2, cudaMemcpyHostToDevice, streams[ i ] );

    i++;
    cudaMemcpyAsync( rgb_d + 3 * rows * cols /2, rgb + 3 * rows * cols /2, size_bytes/2, cudaMemcpyHostToDevice, streams[ i ] );
   // for( std::size_t i = 0 ; i < 2 ; ++i ){
    //    cudaMemcpyAsync( rgb_d + i * 3 * rows * cols /2 - 3 * cols, rgb + i * 3 * rows * cols /2 - 3 * cols, size_bytes/2, cudaMemcpyHostToDevice, streams[ i ] );
    //}

    // cudaMemcpy(rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice);

    dim3 block(32, 4);
    dim3 grid0((cols - 1) / block.x + 1, (rows - 1) / block.y + 1);
    /**
     * Pour la version shared il faut faire superposer les blocs de 2 pixels
     * pour ne pas avoir de bandes non calculées autour des blocs
     * on crée donc plus de blocs.
     */
    dim3 grid1((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure du temps de calcul du kernel uniquement.
    cudaEventRecord(start);

    /*
    // Version en 2 étapes.
    grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
    sobel<<< grid0, block >>>( g_d, s_d, cols, rows );
    */

    /*
    // Version en 2 étapes, Sobel avec mémoire shared.
    grayscale<<< grid0, block >>>( rgb_d, g_d, cols, rows );
    sobel_shared<<< grid1, block, block.x * block.y >>>( g_d, s_d, cols, rows );
    */

    // Version fusionnée.
    // mémoire shared paramètre --> block.x * block.y
    /**
     * for( int i = 0; i < 8; i++){
        flou <<< grid0, block >>> (rgb_d, s_d, cols, rows);
        flou <<< grid0, block >>> (s_d, rgb_d, cols, rows);

    }
     flou <<< grid0, block >>> (rgb_d, s_d, cols, rows);
     */


    dim3 grid2((cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1);


    for( std::size_t i = 0 ; i < 2 ; ++i )
    {
        flou_shared<<< grid2, block, 3 * block.x * block.y, streams[ i ] >>>( rgb_d + i * size/2, s_d + i * size/2, cols, rows/2 );
    }

    // flou_shared<<< grid1, block, 3 * block.x * block.y >>>( rgb_d, s_d, cols, rows );

    cudaMemcpyAsync( g , s_d, size_bytes/2 - 3 * cols, cudaMemcpyDeviceToHost, streams[ 0 ] );
    cudaMemcpyAsync( g + size/2 - 3 * cols, s_d + size/2 + 3 * cols, size_bytes/2 - 3 * cols, cudaMemcpyDeviceToHost, streams[ 1 ] );



    // for( std::size_t i = 0 ; i < 2 ; ++i )
    // {
    //    cudaMemcpyAsync( g + i*size/2, s_d + i * size/2, size_bytes/2, cudaMemcpyDeviceToHost, streams[ i ] );
    // }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    // cudaMemcpy(g, s_d, 3 * rows * cols, cudaMemcpyDeviceToHost);


    cudaEventSynchronize(stop);

    for( std::size_t i = 0 ; i < 2 ; ++i )
    {
        cudaStreamDestroy( streams[ i ] );
    }

    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    std::cout << "time=" << duration << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cv::imwrite(img_out, m_out);

    cudaFree(rgb_d);
    cudaFree(s_d);

    cudaFreeHost(g);
    cudaFreeHost(rgb);

    return 0;
}
