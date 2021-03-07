#include <iostream>
#include "menu_lib.hpp"
#include "utilities.hpp"

#define IMG_IN_PATH "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/in.jpg"
#define IMG_OUT_PATH "/mnt/data/tsky-19/eclipsec/CUDAProjectV2/out.jpg"

#define DEFAULT_DIMX 32
#define DEFAULT_DIMY 4

void printParameters( std::string txtBold, std::string txtNormal, bool isTxtBoldUnderlined )
{
    std::cout << "\033[1" << ((isTxtBoldUnderlined) ? ";4" : "") << "m" << txtBold << "\033[0m" << txtNormal << std::endl;
}

EffectStyle stringToFilterStyle(std::string in){
    if( in.compare("boxblur") == 0 ) return BOXBLUR;
    else if( in.compare("gaussianblur") == 0 ) return GAUSSIANBLUR;
    else if( in.compare("gaussianblur5") == 0 ) return GAUSSIANBLUR5;
    else if( in.compare("emboss") == 0 ) return EMBOSS;
    else if( in.compare("emboss5") == 0 ) return EMBOSS5;
    else if( in.compare("sharpen") == 0 ) return SHARPEN;

    return BOXBLUR;
}
std::string filterStyletoString(EffectStyle style){
    if( style == BOXBLUR ) return "boxblur";
    else if( style == GAUSSIANBLUR ) return "gaussianblur";
    else if( style == GAUSSIANBLUR5 ) return "gaussianblur5";
    else if( style == EMBOSS ) return "emboss";
    else if( style == EMBOSS5 ) return "emboss5";
    else if( style == SHARPEN ) return "sharpen";
    return "boxblur";
}

void printOptionSelection(std::string &img_in_path, std::string &img_out_path, MenuSelection &menuChoice){
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
void presavedParameters( std::string &img_in_path, std::string &img_out_path, MenuSelection &menuChoice )
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
int initParameters( std::string &img_in_path, std::string &img_out_path,
                     MenuSelection &menuChoice, int argc , char **argv )
{
    if( argc < 2){
        presavedParameters(img_in_path, img_out_path, menuChoice);
        printOptionSelection(img_in_path, img_out_path, menuChoice);
        return 0;
    } else if( argc < 2){
        // wrong arguments: Message TODO
        return 1;
    }

    std::cout << std::endl;

    // Retrieve program parameters
    img_in_path = argv[1];
    img_out_path = argv[2];

    if( argc >= 6){

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

        if( argc == 5 ){
            menuChoice.use_shared = std::atoi(argv[5]);
            menuChoice.block.dimY = std::atoi(argv[4]);
            menuChoice.block.dimX = std::atoi(argv[3]);
        }else if( argc == 4){
            menuChoice.use_shared = false;
            menuChoice.block.dimY = std::atoi(argv[4]);
            menuChoice.block.dimX = std::atoi(argv[3]);
        }else if(argc == 3){
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
