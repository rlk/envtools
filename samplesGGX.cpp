#include <iostream>
#include <getopt.h>
#include <cstdio>
#include <cstdlib>

#include "Cubemap"

typedef unsigned int uint;
typedef unsigned char ubyte;

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " outpufile nbsamples mip0Size nbSteps" << std::endl;
    std::cerr << "generate ggx samples by steps excluding level 0" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{

    uint mip0Size = 0;
    uint nbSteps = 0;
    uint samples = 4096;

    if ( argc < 5 )
        return usage(argv[0]);

    std::string output = std::string( argv[1] );
    samples = atoi( argv[2] );
    mip0Size = atoi( argv[3] );
    nbSteps = atoi( argv[4] ) + 1;

    float step = 1.0/(nbSteps-1.0);

    FILE* file = fopen(output.c_str(),"wb");
    std::cout << "compute " << nbSteps << " levels sample GGX from roughness  " << step << " to 1.0" << std::endl;
    for ( uint i = 1; i < nbSteps; i++ ) {

        float r = step * i;
        float roughnessLinear = r; //pow(r,1.5);
        precomputedLightInLocalSpace( samples, roughnessLinear, mip0Size );

        ubyte* buffer = (ubyte*)getPrecomputedLightCache();
        fwrite(buffer, samples*4*4, 1 , file );

    }

    return 0;
}
