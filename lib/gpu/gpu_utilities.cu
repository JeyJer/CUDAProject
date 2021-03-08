#include <iostream>
#include "gpu_utilities.cuh"

void GpuUtilMenuSelection::printOptionSelection(std::string &img_in_path, std::string &img_out_path,
        GpuUtilMenuSelection &menuChoice){
    printParameters( "• Files path :", "", false );
    printParameters( "In path :", " "+(img_in_path), true );
    printParameters( "Out path :", " "+(img_out_path), true );
    std::cout << std::endl;

    printParameters( "• CUDA Options :", "", false );
    printParameters( "Block dim X :", " "+std::to_string(menuChoice.block.dimX), true );
    printParameters( "Block dim Y :", " "+std::to_string(menuChoice.block.dimY), true );
    printParameters( "Memory Shared enabled ?", ((menuChoice.use_shared) ? " Yes" : " No"), true );
    if( menuChoice.nb_stream > 0 )
        printParameters( "Streams enabled :", " "+std::to_string(menuChoice.nb_stream), true );
    else
        printParameters( "Streams enabled ?", " No", true );
    std::cout << std::endl;

    printParameters( "• Image filters :", "", false );
    for( int i = 0 ; i < menuChoice.enabled_filters.size() ; ++i )
        printParameters(  filterStyletoString(menuChoice.enabled_filters.at(i)) + " :",
                " "+std::to_string(menuChoice.nb_pass.at(i)) + "-pass.", true );

    std::cout << std::endl;

}
void GpuUtilMenuSelection::presavedParameters(std::string &img_in_path, std::string &img_out_path,
        GpuUtilMenuSelection &menuChoice )
{
    img_in_path = IMG_IN_PATH;
    img_out_path = IMG_OUT_PATH;
    menuChoice.block.dimX = DEFAULT_DIMX;
    menuChoice.block.dimY = DEFAULT_DIMY;
    menuChoice.use_shared = false;
    menuChoice.nb_stream = 0;
    menuChoice.enabled_filters.push_back(BOXBLUR);
    menuChoice.nb_pass.push_back( 1 );
}
int GpuUtilMenuSelection::initParameters(std::string &img_in_path,
                                         std::string &img_out_path, GpuUtilMenuSelection &menuChoice,
                                         int argc, char **argv)
{
    if( argc < 2){
        presavedParameters(img_in_path, img_out_path, menuChoice);
        printOptionSelection(img_in_path, img_out_path, menuChoice);
        return 0;
    } else if( argc < 3){
        // wrong arguments: Message TODO
        return 1;
    }

    std::cout << std::endl;

    // Retrieve program parameters
    img_in_path = argv[1];
    img_out_path = argv[2];

    if( argc >= 7){

        menuChoice.block.dimX = std::atoi(argv[3]);
        menuChoice.block.dimY = std::atoi(argv[4]);

        // Save if the program will use shared memory.
        menuChoice.use_shared = std::atoi( argv[5] );
        menuChoice.nb_stream = std::atoi( argv[6] );

        for( int i = 7 ; i < argc ; i+=2 )
        {
            menuChoice.enabled_filters.push_back( stringToFilterStyle(argv[i]) );
            menuChoice.nb_pass.push_back( std::atoi(argv[i+1]) );
        }

    } else {
        menuChoice.enabled_filters.push_back(BOXBLUR);
        menuChoice.nb_pass.push_back( 1 );

        if( argc == 6 ){
            menuChoice.use_shared = std::atoi(argv[5]);
            menuChoice.block.dimY = std::atoi(argv[4]);
            menuChoice.block.dimX = std::atoi(argv[3]);
        }else if( argc == 5){
            menuChoice.use_shared = false;
            menuChoice.block.dimY = std::atoi(argv[4]);
            menuChoice.block.dimX = std::atoi(argv[3]);
        }else if(argc == 4){
            menuChoice.use_shared = false;
            menuChoice.block.dimY = DEFAULT_DIMY;
            menuChoice.block.dimX = std::atoi(argv[3]);
        } else {
            menuChoice.use_shared = false;
            menuChoice.block.dimY = DEFAULT_DIMY;
            menuChoice.block.dimX = DEFAULT_DIMX;
        }
    }
    printOptionSelection(img_in_path, img_out_path, menuChoice);
    return 0;
}