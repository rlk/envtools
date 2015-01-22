#include <iostream>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>

#include "Cubemap"

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-s size] [-n nbsamples] [-r roughness ] [-f toggle fixup edge ] in.tif out.tif" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{

    int size = 0;
    int c;
    int samples = 128;
    int fixup = 0;
    float roughnessLinear = 0.1;

    while ((c = getopt(argc, argv, "s:n:r:f")) != -1)
        switch (c)
        {
        case 's': size = atoi(optarg);       break;
        case 'n': samples = atoi(optarg);  break;
        case 'r': roughnessLinear = atof(optarg);  break;
        case 'f': fixup = 1;  break;

        default: return usage(argv[0]);
        }

    std::string input, output;
    if ( optind < argc-1 ) {

        // generate specular ibl
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );

        Cubemap image;
        image.loadCubemap(input);
        image.computeBackground( output, size, samples, roughnessLinear, fixup );

    } else {
        return usage( argv[0] );
    }


    return 0;
}
