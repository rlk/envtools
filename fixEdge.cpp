#include "Cubemap"
#include <iostream>

int main( int argc, char** argv )
{
    if (argc != 3 ) {
        std::cout << "usage " << argv[0] << " input " << " level" << std::endl;
        return 1;
    }

    Cubemap cubemap;

    std::string input = std::string ( argv[1] );
    std::string level = std::string ( argv[2] );
    std::string output = "./";

    std::size_t pos = input.rfind("/");
    if ( pos != std::string::npos )
        output = input.substr( 0, pos + 1);

    std::cout << "fixEdge " << input << " to " << output  << " at " << level << " level " << std::endl;

    cubemap.loadCubemap( input );
    cubemap.fixupCubeEdges( output, atof(level.c_str())  );

    return 0;
}
