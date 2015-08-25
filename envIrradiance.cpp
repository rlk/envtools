#include <iostream>
#include <cstdlib>
#include <getopt.h>

#include "Cubemap"

static int usage(const char *exe)
{
    std::cerr << "Usage: " << exe <<  " [-n n] [-f f] in.tif out.tif\n" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{
    int     n = 256;
    int     c;
    std::string fixupString;

    while ((c = getopt(argc, argv, "n:")) != -1)
        switch (c)
        {
            case 'n': n = strtol(optarg, 0, 0);       break;
            case 'f': fixupString = optarg;   break;
            default:
                return usage(argv[0]);
        }


    std::string input, output;
    int fixup = 0;

    if ( fixupString == "stretch" ) {
        fixup = 1;
    }

    if ( optind < argc - 1)
    {
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );

        Cubemap cubemap;
        cubemap.load(input);
        Cubemap* result = cubemap.shFilterCubeMap( true, fixup, n );
        result->write(output);
        delete result;
    } else {
        return usage(argv[0]);
    }

    return 0;
}
