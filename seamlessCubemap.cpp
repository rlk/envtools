#include "Cubemap"
#include "cstdlib"
#include <iostream>

int main( int argc, char** argv )
{
    if (argc < 2 ) {
        std::cout << "usage " << argv[0] << " input output" << std::endl;
        return 1;
    }

    Cubemap cubemap;

    std::string input = std::string ( argv[1] );
    std::string output = "./";

    std::size_t pos = input.rfind("/");
    if ( pos != std::string::npos )
        output = input.substr( 0, pos + 1);

    if ( argc > 2 ) {
        output = argv[2];
    }

    std::cout << argv[0] << " " << input << " to " << output  << std::endl;

    cubemap.loadCubemap( input );
    Cubemap* seamless = cubemap.makeSeamless();
    seamless->write( output );

    return 0;
}
