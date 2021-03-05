#include <opencv2/opencv.hpp>
#include <vector>
#include <cstring>

//----------------
//--- UTILITY ----
//----------------

//-- PARAMETERS --

void printParameters( std::string txtBold, std::string txtNormal, bool isTxtBoldUnderlined )
{
    std::cout << "\033[1" << ((isTxtBoldUnderlined) ? ";4" : "") << "m" << txtBold << "\033[0m" << txtNormal << std::endl;
}

void initParameters( std::string * img_in_path, std::string * img_out_path,
    int * blockX, int * blockY, bool * useShared, int * streamNbr,
    std::vector<std::string> * filtersEnabled, std::vector<int> * passNumber,
    int argc , char **argv )
{
    std::cout << std::endl;

    // Retrieve program parameters
    *img_in_path = argv[1];
    *img_out_path = argv[2];

    *blockX = std::atoi(argv[3]);
    *blockY = std::atoi(argv[4]);

    // Save if the program will use shared memory.
    *useShared = std::atoi( argv[5] );
    *streamNbr = std::atoi( argv[6] );

    for( int i = 7 ; i < argc ; i+=2 )
    {
        filtersEnabled->push_back( argv[i] );
        passNumber->push_back( std::atoi(argv[i+1]) );
    }

    printParameters( "• Files path :", "", false );
    printParameters( "In path :", " "+(*img_in_path), true );
    printParameters( "Out path :", " "+(*img_out_path), true );
    std::cout << std::endl;

    printParameters( "• CUDA Options :", "", false );
    printParameters( "Block dim X :", " "+std::to_string(*blockX), true );
    printParameters( "Block dim Y :", " "+std::to_string(*blockY), true );
    printParameters( "Memory Shared enabled ?", ((*useShared) ? " Yes" : " No"), true );
    if( *streamNbr > 0 )
        printParameters( "Streams enabled :", " "+std::to_string(*streamNbr), true );
    else
        printParameters( "Streams enabled ?", " No", true );
    std::cout << std::endl;

    printParameters( "• Image filters :", "", false );
    for( int i = 0 ; i < filtersEnabled->size() ; ++i )
    {
        printParameters( filtersEnabled->at(i) + " :", " "+std::to_string(passNumber->at(i)) + "-pass.", true );
    }
    std::cout << std::endl;
}

void presavedParameters( std::string * img_in_path, std::string * img_out_path,
    int * blockX, int * blockY, bool * useShared, int * streamNbr,
    std::vector<std::string> * filtersEnabled, std::vector<int> * passNumber )
{
    *img_in_path = "./in.jpg";
    *img_out_path = "./out.jpg";
    *blockX = 4;
    *blockY = 32;
    *useShared = 0;
    *streamNbr = 0;
    filtersEnabled->push_back( "BoxBlur" );
    passNumber->push_back( 10 );
}

//---- FILTERS ---

int init_divider( std::string filter )
{
    if( filter.compare("boxblur") == 0 )
    {
        return 9;
    }
    else if( filter.compare("gaussianblur") == 0 )
    {
        return 16;
    }
    else
    {
        return 1;
    }
}

void init_edge_detection_matrix( char * conv_matrix_h )
{
    conv_matrix_h[0] = -1;
    conv_matrix_h[1] = -1;
    conv_matrix_h[2] = -1;
    conv_matrix_h[3] = -1;
    conv_matrix_h[4] = 8;
    conv_matrix_h[5] = -1;
    conv_matrix_h[6] = -1;
    conv_matrix_h[7] = -1;
    conv_matrix_h[8] = -1;
}

void init_sharpen_matrix( char * conv_matrix_h )
{
    conv_matrix_h[0] = 0;
    conv_matrix_h[1] = -1;
    conv_matrix_h[2] = 0;
    conv_matrix_h[3] = -1;
    conv_matrix_h[4] = 5;
    conv_matrix_h[5] = -1;
    conv_matrix_h[6] = 0;
    conv_matrix_h[7] = -1;
    conv_matrix_h[8] = 0;
}

void init_box_blur_matrix( char * conv_matrix_h )
{
    conv_matrix_h[0] = 1;
    conv_matrix_h[1] = 1;
    conv_matrix_h[2] = 1;
    conv_matrix_h[3] = 1;
    conv_matrix_h[4] = 1;
    conv_matrix_h[5] = 1;
    conv_matrix_h[6] = 1;
    conv_matrix_h[7] = 1;
    conv_matrix_h[8] = 1;
}

void init_gaussian_blur_matrix( char * conv_matrix_h )
{
    conv_matrix_h[0] = 1;
    conv_matrix_h[1] = 2;
    conv_matrix_h[2] = 1;
    conv_matrix_h[3] = 2;
    conv_matrix_h[4] = 4;
    conv_matrix_h[5] = 2;
    conv_matrix_h[6] = 1;
    conv_matrix_h[7] = 2;
    conv_matrix_h[8] = 1;
}

void init_emboss_matrix( char * conv_matrix_h )
{
    conv_matrix_h[0] = -2;
    conv_matrix_h[1] = -1;
    conv_matrix_h[2] = 0;
    conv_matrix_h[3] = -1;
    conv_matrix_h[4] = 1;
    conv_matrix_h[5] = 1;
    conv_matrix_h[6] = 0;
    conv_matrix_h[7] = 1;
    conv_matrix_h[8] = 2;
}

void init_conv_matrix_h( std::string filter, char * conv_matrix_h )
{
    if( filter.compare("edgedetection") == 0 )
    {
        init_edge_detection_matrix( conv_matrix_h );
    }
    else if( filter.compare("sharpen") == 0 )
    {
        init_sharpen_matrix( conv_matrix_h );
    }
    else if( filter.compare("boxblur") == 0 )
    {
        init_box_blur_matrix( conv_matrix_h );
    }
    else if( filter.compare("gaussianblur") == 0 )
    {
        init_gaussian_blur_matrix( conv_matrix_h );
    }
    else if( filter.compare("emboss") == 0 )
    {
        init_emboss_matrix( conv_matrix_h );
    }
    else
    {
        conv_matrix_h = nullptr;
    }
}

//---- POINTER MANIPULATION ----

void invert_pointer( unsigned char ** ptr1, unsigned char ** ptr2 )
{
    unsigned char* invertion_ptr = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = invertion_ptr;
}

//----------------
//----- CUDA -----
//----------------

//---- CHRONO ----

void initCudaChrono( cudaEvent_t * start, cudaEvent_t * stop )
{
    cudaEventCreate( start );
    cudaEventCreate( stop );
}

void recordCudaChrono( cudaEvent_t * chrono )
{
    cudaEventRecord( *chrono );
}

float getCudaChronoTimeElapsed( cudaEvent_t * start, cudaEvent_t * stop )
{
    float duration;
    cudaEventElapsedTime( &duration, *start, *stop );
    return duration;
}

void destroyCudaChrono( cudaEvent_t * start, cudaEvent_t * stop )
{
    cudaEventDestroy( *start );
    cudaEventDestroy( *stop );
}

//---- PROCESSING ----

__global__ void image_processing(unsigned char* rgb, unsigned char* s, std::size_t cols, std::size_t rows, char * matrix, int divider )
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < cols && j > 0 && j < rows)
    {
        auto h_r = matrix[0] * rgb[3 * ((j - 1) * cols + i - 1)] + matrix[1] * rgb[3 * ((j - 1) * cols + i)] + matrix[2] * rgb[3 * ((j - 1) * cols + i + 1)]
                 + matrix[3] * rgb[3 * ((j    ) * cols + i - 1)] + matrix[4] * rgb[3 * ((j    ) * cols + i)] + matrix[5] * rgb[3 * ((j    ) * cols + i + 1)]
                 + matrix[6] * rgb[3 * ((j + 1) * cols + i - 1)] + matrix[7] * rgb[3 * ((j + 1) * cols + i)] + matrix[8] * rgb[3 * ((j + 1) * cols + i + 1)];

        auto h_g = matrix[0] * rgb[3 * ((j - 1) * cols + i - 1) + 1] + matrix[1] * rgb[3 * ((j - 1) * cols + i) + 1] + matrix[2] * rgb[3 * ((j - 1) * cols + i + 1) + 1]
                 + matrix[3] * rgb[3 * ((j    ) * cols + i - 1) + 1] + matrix[4] * rgb[3 * ((j    ) * cols + i) + 1] + matrix[5] * rgb[3 * ((j    ) * cols + i + 1) + 1]
                 + matrix[6] * rgb[3 * ((j + 1) * cols + i - 1) + 1] + matrix[7] * rgb[3 * ((j + 1) * cols + i) + 1] + matrix[8] * rgb[3 * ((j + 1) * cols + i + 1) + 1];

        auto h_b = matrix[0] * rgb[3 * ((j - 1) * cols + i - 1) + 2] + matrix[1] * rgb[3 * ((j - 1) * cols + i) + 2] + matrix[2] * rgb[3 * ((j - 1) * cols + i + 1) + 2]
                 + matrix[3] * rgb[3 * ((j    ) * cols + i - 1) + 2] + matrix[4] * rgb[3 * ((j    ) * cols + i) + 2] + matrix[5] * rgb[3 * ((j    ) * cols + i + 1) + 2]
                 + matrix[6] * rgb[3 * ((j + 1) * cols + i - 1) + 2] + matrix[7] * rgb[3 * ((j + 1) * cols + i) + 2] + matrix[8] * rgb[3 * ((j + 1) * cols + i + 1) + 2];

        s[3 * (j * cols + i)    ] = (h_r / divider);
        s[3 * (j * cols + i) + 1] = (h_g / divider);
        s[3 * (j * cols + i) + 2] = (h_b / divider);
    }
}

__global__ void image_processing_shared(unsigned char* rgb, unsigned char* s, std::size_t cols, std::size_t rows, char * matrix, int divider)
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
        sh[3 * (j * w + i)    ] = rgb[3 * (j_global * cols + i_global)    ];
        sh[3 * (j * w + i) + 1] = rgb[3 * (j_global * cols + i_global) + 1];
        sh[3 * (j * w + i) + 2] = rgb[3* ( j_global * cols + i_global) + 2];
    }

    __syncthreads();

    if (i_global < cols - 1 && j_global < rows - 1 && i > 0 && i < (w - 1) && j > 0 && j < (height - 1))
    {
        auto h_r = matrix[0] * sh[3 * ((j - 1) * w + i - 1)] + matrix[1] * sh[3 * ((j - 1) * w + i)] + matrix[2] * sh[3 * ((j - 1) * w + i + 1)]
                 + matrix[3] * sh[3 * ((j    ) * w + i - 1)] + matrix[4] * sh[3 * ((j    ) * w + i)] + matrix[5] * sh[3 * ((j    ) * w + i + 1)]
                 + matrix[6] * sh[3 * ((j + 1) * w + i - 1)] + matrix[7] * sh[3 * ((j + 1) * w + i)] + matrix[8] * sh[3 * ((j + 1) * w + i + 1)];

        auto h_g = matrix[0] * sh[3 * ((j - 1) * w + i - 1) + 1] + matrix[1] * sh[3 * ((j - 1) * w + i) + 1] + matrix[2] * sh[3 * ((j - 1) * w + i + 1) + 1]
                 + matrix[3] * sh[3 * ((j    ) * w + i - 1) + 1] + matrix[4] * sh[3 * ((j    ) * w + i) + 1] + matrix[5] * sh[3 * ((j    ) * w + i + 1) + 1]
                 + matrix[6] * sh[3 * ((j + 1) * w + i - 1) + 1] + matrix[7] * sh[3 * ((j + 1) * w + i) + 1] + matrix[8] * sh[3 * ((j + 1) * w + i + 1) + 1];

        auto h_b = matrix[0] * sh[3 * ((j - 1) * w + i - 1) + 2] + matrix[1] * sh[3 * ((j - 1) * w + i) + 2] + matrix[2] * sh[3 * ((j - 1) * w + i + 1) + 2]
                 + matrix[3] * sh[3 * ((j    ) * w + i - 1) + 2] + matrix[4] * sh[3 * ((j    ) * w + i) + 2] + matrix[5] * sh[3 * ((j    ) * w + i + 1) + 2]
                 + matrix[6] * sh[3 * ((j + 1) * w + i - 1) + 2] + matrix[7] * sh[3 * ((j + 1) * w + i) + 2] + matrix[8] * sh[3 * ((j + 1) * w + i + 1) + 2];

        s[3 * (j_global * cols + i_global)    ] = (h_r / divider);
        s[3 * (j_global * cols + i_global) + 1] = (h_g / divider);
        s[3 * (j_global * cols + i_global) + 2] = (h_b / divider);
    }
}

//----------------
//----- MAIN -----
//----------------

int main( int argc , char **argv )
{
    //---- Declarate and allocate parameters
    std::string img_in_path, img_out_path;
    int blockX, blockY, streamNbr;
    bool useShared;
    std::vector<std::string> filtersEnabled;
    std::vector<int> passNumber;

    //---- Initialize parameters
    // RELEASE_MODE
    initParameters( &img_in_path, &img_out_path, &blockX, &blockY, &useShared, &streamNbr, &filtersEnabled, &passNumber, argc, argv );
    // DEBUG_MODE
    // presavedParameters( &img_in_path, &img_out_path, &blockX, &blockY, &useShared, &streamNbr, &filtersEnabled, &passNumber );

    //---- Retrieve image properties
    cv::Mat img_in_matrix = cv::imread( img_in_path, cv::IMREAD_UNCHANGED );
    auto rows = img_in_matrix.rows;
    auto cols = img_in_matrix.cols;

    //---- Allocate a cv::Mat (host-side) to store the device result
    unsigned char* img_out_h = nullptr;
    cudaMallocHost( &img_out_h, 3 * rows * cols );
    cv::Mat img_out_matrix( rows, cols, CV_8UC3, img_out_h );

    //---- allocate and initialize image's pixel array (host-side)
    unsigned char* rgb = nullptr;
    cudaMallocHost( &rgb, 3 * rows * cols );
    std::memcpy( rgb, img_in_matrix.data, 3 * rows * cols );

    //---- allocate and initialize image's pixel array (device-side)
    unsigned char* rgb_d;
    unsigned char* result_d;
    cudaMalloc( &rgb_d, 3 * rows * cols );
    cudaMalloc( &result_d, 3 * rows * cols );
    cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );

    //---- Threads distribution
    // grid block
    dim3 block( blockX, blockY );
    // grid for non-shared memory processing
    dim3 grid0( (cols - 1) / block.x + 1, (rows - 1) / block.y + 1 );
    // grid for shared memory processing
    dim3 grid1( (cols - 1) / (block.x - 2) + 1, (rows - 1) / (block.y - 2) + 1 );

    //---- Init and start chrono
    cudaEvent_t start, stop;
    initCudaChrono( &start, &stop );

    //---- Launch image processing loop
    for( int i = 0 ; i < filtersEnabled.size() ; ++i )
    {
        // init convolution matrix and divider according to the filter selected
        char * conv_matrix_h, *conv_matrix_d;
        cudaMallocHost( &conv_matrix_h, sizeof(char)*9 );
        init_conv_matrix_h( filtersEnabled.at(i), conv_matrix_h );
        cudaMalloc( &conv_matrix_d, sizeof(char)*9 );
        cudaMemcpy( conv_matrix_d, conv_matrix_h, sizeof(char)*9, cudaMemcpyHostToDevice );

        int divider = init_divider( filtersEnabled.at(i) );

        // apply the filter how many passes wished
        for( int j = 0 ; j < passNumber.at(i) ; ++j )
        {
            recordCudaChrono( &start );
            if( !useShared )
            {
                image_processing<<< grid0, block >>>( rgb_d, result_d, cols, rows, conv_matrix_d, divider );

                cudaDeviceSynchronize();
                cudaError err = cudaGetLastError();
                if( err != cudaSuccess )
                    std::cerr << cudaGetErrorString( err ) << std::endl;
            }
            else
            {
                image_processing_shared<<< grid1, block, 3 * block.x * block.y >>>( rgb_d, result_d, cols, rows, conv_matrix_d, divider );

                cudaDeviceSynchronize();
                cudaError err = cudaGetLastError();
                if( err != cudaSuccess )
                    std::cerr << cudaGetErrorString( err ) << std::endl;
            }
            //---- get chrono time elapsed
            recordCudaChrono( &stop );
            cudaEventSynchronize( stop );
            float duration = getCudaChronoTimeElapsed( &start, &stop );
            std::cout << "Pass duration : " << duration << "ms" << std::endl;
            // TODO Do something with duration

            // invert rgb_d with result_d, for any other pass
            invert_pointer( &rgb_d, &result_d );
        }
        // TODO free matrix here
        cudaFreeHost( conv_matrix_h );
        cudaFree( conv_matrix_d );
    }
    // cancel the rgb_d and result_d invertion, to put back the result in result_d
    invert_pointer( &rgb_d, &result_d );

    //---- Copy the result to cv::Mat
    cudaMemcpy( img_out_h, result_d, 3 * rows * cols, cudaMemcpyDeviceToHost );

    //---- Write img_out onto the disk
    cv::imwrite( cv::String(img_out_path), img_out_matrix );

    //---- Free memory
    // host-side
    cudaFree( rgb_d );
    cudaFree( result_d );
    // device-side
    cudaFreeHost( img_out_h );
    cudaFreeHost( rgb );
    destroyCudaChrono( &start, &stop );

    return 0;
}
