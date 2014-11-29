#include <iostream>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>

#include "Cubemap"

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-s size] [-m minMipmap] [-n nbsamples] in.tif out.tif" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{

    int size = 0;
    int c;
    int minMipmap = 0;
    int samples = 1024;

    while ((c = getopt(argc, argv, "s:m:n:")) != -1)
        switch (c)
        {
        case 's': size = atof(optarg);       break;
        case 'm': minMipmap = atof(optarg);  break;
        case 'n': samples = atoi(optarg);  break;

        default: return usage(argv[0]);
        }

    std::string input, output;
    if ( optind < argc-1 ) {

        // generate specular ibl
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );

        Cubemap image;
        image.loadCubemap(input);
        image.computePrefilteredEnvironment( output, size, minMipmap, samples );

    } else {
        return usage( argv[0] );
    }


    return 0;
}
