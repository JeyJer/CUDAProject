#include "cpu_utilities.hpp"

void CpuUtilMenuSelection::printOptionSelection(std::string &img_in_path, std::string &img_out_path,
                                                CpuUtilMenuSelection &menuChoice){
    printParameters( "• Files path :", "", false );
    printParameters( "In path :", " "+(img_in_path), true );
    printParameters( "Out path :", " "+(img_out_path), true );
    std::cout << std::endl;

    printParameters( "• CPU Options :", "", false );
    std::cout << std::endl;

    printParameters( "• Image filters :", "", false );
    for( int i = 0 ; i < menuChoice.enabled_filters.size() ; ++i )
        printParameters(  filterStyletoString(menuChoice.enabled_filters.at(i)) + " :",
                          " "+std::to_string(menuChoice.nb_pass.at(i)) + "-pass.", true );

    std::cout << std::endl;

}
void CpuUtilMenuSelection::presavedParameters(std::string &img_in_path, std::string &img_out_path,
                                              CpuUtilMenuSelection &menuChoice )
{
    img_in_path = IMG_IN_PATH;
    img_out_path = IMG_OUT_PATH;

    menuChoice.enabled_filters.push_back(BOXBLUR);
    menuChoice.nb_pass.push_back( 1 );
}
int CpuUtilMenuSelection::initParameters(std::string &img_in_path, std::string &img_out_path,
        CpuUtilMenuSelection &menuChoice, int argc, char **argv) {
    if( argc < 2){
        CpuUtilMenuSelection::presavedParameters(img_in_path, img_out_path, menuChoice);
        CpuUtilMenuSelection::printOptionSelection(img_in_path, img_out_path, menuChoice);
        return 0;
    } else if( argc < 3){
        // wrong arguments: Message TODO
        return 1;
    }

    std::cout << std::endl;

    // Retrieve program parameters
    img_in_path = argv[1];
    img_out_path = argv[2];

    if( argc >= 4) {

        for (int i = 3; i < argc; i += 2) {
            menuChoice.enabled_filters.push_back(stringToFilterStyle(argv[i]));
            menuChoice.nb_pass.push_back(std::atoi(argv[i + 1]));
        }
    } else {
        menuChoice.enabled_filters.push_back(BOXBLUR);
        menuChoice.nb_pass.push_back( 1 );
    }

    CpuUtilMenuSelection::printOptionSelection(img_in_path, img_out_path, menuChoice);
    return 0;


}
