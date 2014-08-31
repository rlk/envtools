#include <iostream>
#include <getopt.h>

#include "Cubemap"

static int usage(const char *exe)
{
    std::cerr << "Usage: " << exe <<  " [-n n] in.tif out.tif\n" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{
    int     n = 256;
    int     c;

    while ((c = getopt(argc, argv, "n:")) != -1)
        switch (c)
        {
            case 'n': n = strtol(optarg, 0, 0);       break;
            default:
                return usage(argv[0]);
        }

    std::string input, output;

    if ( optind < argc - 1)
    {
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );

        Cubemap cubemap;
        cubemap.loadCubemap(input);
        Cubemap* result = cubemap.shFilterCubeMap( true, FIXUP_NONE, n );
        result->write(output);
        delete result;
    } else {
        return usage(argv[0]);
    }

    return 0;
}
