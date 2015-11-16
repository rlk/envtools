#include <iostream>
#include <getopt.h>
#include <cstdlib>
#include <sstream>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "Math"
#include "Cubemap"
#include "Color"



static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " input " << std::endl;
    std::cerr << "eg: " << name << "input_%i.tiff " << std::endl;

    return 1;
}

// mipmapped cubemap only
int main(int argc, char** argv)
{

    std::string input;

    int c;

    while ((c = getopt (argc, argv, "")) != -1);

    if ( optind < argc ) {

        input = std::string( argv[optind] );

        Cubemap elight;

        if ( input.find("%") != std::string::npos )
            elight.loadMipMap(input);
        else
            elight.load(input);

        elight.computeMainLightDirection ();

    } else {
        return usage( argv[0] );
    }

    return 0;
}
