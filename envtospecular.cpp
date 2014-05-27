/* Copyright (c) 2014 Cedric Pinson                                           */
/*                                                                            */
/* Permission is hereby granted, free of charge, to any person obtaining a    */
/* copy of this software and associated documentation files (the "Software"), */
/* to deal in the Software without restriction, including without limitation  */
/* the rights to use, copy, modify, merge, publish, distribute, sublicense,   */
/* and/or sell copies of the Software, and to permit persons to whom the      */
/* Software is furnished to do so, subject to the following conditions:       */
/*                                                                            */
/* The above copyright notice and this permission notice shall be included in */
/* all copies or substantial portions of the Software.                        */
/*                                                                            */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    */
/* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    */
/* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        */
/* DEALINGS IN THE SOFTWARE.                                                  */


#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <getopt.h>

typedef unsigned int uint;

#include <tiffio.h>

#include "gray.h"
#include "sRGB.h"

#include "Math"


static Vec3d cubemap_face[6][3] = {
    { Vec3d(0,0,-1), Vec3d(0,-1,0), Vec3d(1,0,0) },// x positif
    { Vec3d(0,0,1), Vec3d(0,-1,0), Vec3d(-1,0,0) }, // x negatif

    { Vec3d(1,0,0), Vec3d(0,0,1), Vec3d(0,1,0) },  // y positif
    { Vec3d(1,0,0), Vec3d(0,0,-1),Vec3d(0,-1,0) }, // y negatif

    { Vec3d(1,0,0), Vec3d(0,-1,0), Vec3d(0,0,1) },  // z positif
    { Vec3d(-1,0,0), Vec3d(0,-1,0),Vec3d(0,0,-1) } // z negatif
};

struct CubemapLevel {
    uint16 _width;
    uint16 _height;
    float* _images[6];
    uint16 _samplePerPixel;
    uint16 _bitsPerPixel;

    CubemapLevel() {
        for ( int i = 0; i < 6; i++ ) {
            _images[i] = 0;
        }
    }
    ~CubemapLevel() {
        for ( int i = 0; i < 6; i++ ) {
            if ( _images[i] )
                delete [] _images[i];
        }
    }

    void init( uint16 width, uint16 height, uint16 sample, uint16 bits) {
        _width = width;
        _height = height;
        _samplePerPixel = sample;
        _bitsPerPixel = bits;

        for ( int i = 0; i < 6; i++ ) {
            _images[i] = new float[width*height*sample];
        }
    }

    void write( const std::string& filename ) {

        TIFF* file = TIFFOpen(filename.c_str(), "w");

        if (!file)
            return;

        for ( int face = 0; face < 6; face++) {
            TIFFSetField(file, TIFFTAG_IMAGEWIDTH,      _width);
            TIFFSetField(file, TIFFTAG_IMAGELENGTH,     _height);
            TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, _samplePerPixel);
            TIFFSetField(file, TIFFTAG_BITSPERSAMPLE,   32);
            TIFFSetField(file, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
            TIFFSetField(file, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
            TIFFSetField(file, TIFFTAG_SAMPLEFORMAT,    SAMPLEFORMAT_IEEEFP);

            if (_samplePerPixel == 1)
            {
                TIFFSetField(file, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
                TIFFSetField(file, TIFFTAG_ICCPROFILE, sizeof (grayicc), grayicc);
            }
            else
            {
                TIFFSetField(file, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_RGB);
                TIFFSetField(file, TIFFTAG_ICCPROFILE, sizeof (sRGBicc), sRGBicc);
            }

            size_t size = (size_t) TIFFScanlineSize( file );

            for (int i = 0; i < _height; ++i) {
                uint8* start = (uint8*)_images[face];
                TIFFWriteScanline(file, (uint8 *) start + size * i, i, 0);
            }

            TIFFWriteDirectory(file);
        }

        TIFFClose(file);
    }


    void iterateOnFace( int face, float roughness, const CubemapLevel& cubemap ) {

        double xInvFactor = 2.0/double(_width);
        #pragma omp parallel
        #pragma omp for
        for ( uint32 j = 0; j < _height; j++ ) {
            int lineIndex = j*_samplePerPixel*_width;

            for ( uint32 i = 0; i < _width; i++ ) {
                int index = lineIndex + i*_samplePerPixel;
                //Vec3d color = Vec3d( _images[face][ index ], _images[face][ index + 1 ], _images[face][ index +2 ] );

                // center ray on texel center
                // generate a vector for each texel
                double texelX = double(i) + 0.5;
                double texelY = double(j) + 0.5;
                Vec3d vecX = cubemap_face[face][0] * (texelX*xInvFactor - 1.0);
                Vec3d vecY = cubemap_face[face][1] * (texelY*xInvFactor - 1.0);
                Vec3d vecZ = cubemap_face[face][2];
                Vec3d direction = Vec3d( vecX + vecY + vecZ );
                direction.normalize();

                Vec3d resultColor = cubemap.prefilterEnvMap( roughness, direction );

                _images[face][ index     ] = resultColor[0];
                _images[face][ index + 1 ] = resultColor[1];
                _images[face][ index + 2 ] = resultColor[2];

                //std::cout << "face " << face << " processing " << i << "x" << j << std::endl;
#if 0
                //sample( direction, resultColor );
                Vec3d diff = (color - resultColor);
                if ( fabs(diff[0]) > 1e-6 || fabs(diff[1]) > 1e-6 || fabs(diff[2]) > 1e-6 ) {
                    std::cout << "face " << face << " " << i << "x" << j << " color error " << diff[0] << " " << diff[1] << " " << diff[2] << std::endl;
                    std::cout << "direction " << direction[0] << " " << direction[1] << " " << direction[2]  << std::endl;
                    return;
                }
#endif

            }
        }
    }

    Vec3d prefilterEnvMap( float roughness, const Vec3d& R ) const {
        Vec3d N = R;
        Vec3d V = R;
        Vec3d prefilteredColor = Vec3d(0,0,0);
        const uint NumSamples = 1024;
        double totalWeight = 0;

        for( uint i = 0; i < NumSamples; i++ ) {
            Vec2d Xi = hammersley( i, NumSamples );
            Vec3d H = importanceSampleGGX( Xi, roughness, N );
            Vec3d L =  H * dot( V, H ) * 2.0 - V;
            double NoL = std::max( 1.0, dot( N, L ) );

            if( NoL > 0 ) {
                Vec3d color;
                sample( L, color );
                prefilteredColor += color * NoL;
                totalWeight += NoL;
            }
        }
        return prefilteredColor / totalWeight;
    }

// major axis
// direction     target                              sc     tc    ma
// ----------    ---------------------------------   ---    ---   ---
//  +rx          GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT   -rz    -ry   rx
//  -rx          GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT   +rz    -ry   rx
//  +ry          GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT   +rx    +rz   ry
//  -ry          GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT   +rx    -rz   ry
//  +rz          GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT   +rx    -ry   rz
//  -rz          GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT   -rx    -ry   rz
// s   =   ( sc/|ma| + 1 ) / 2
// t   =   ( tc/|ma| + 1 ) / 2

    void sample(const Vec3d& direction, Vec3d& color ) const {


        int bestAxis = 0;
        if ( fabs(direction[1]) > fabs(direction[0]) ) {
            bestAxis = 1;
            if ( fabs(direction[2]) > fabs(direction[1]) )
                bestAxis = 2;
        } else if ( fabs(direction[2]) > fabs(direction[0]) )
            bestAxis = 2;

        // select the index of cubemap face
        int index = bestAxis*2 + ( direction[bestAxis] > 0 ? 0 : 1 );
        double bestAxisValue = direction[bestAxis];
        double denom = fabs( bestAxisValue );
        double maInv = 1.0/denom;

        double sc = cubemap_face[index][0] * direction;
        double tc = cubemap_face[index][1] * direction;
        double ppx = (sc * maInv + 1.0) * 0.5 * _width; // width == height
        double ppy = (tc * maInv + 1.0) * 0.5 * _width; // width == height

        //int px = int( floor( ppx +0.5 ) ); // center pixel
        //int py = int( floor( ppy +0.5 ) ); // center pixel

        int px = int( floor( ppx ) ); // center pixel
        int py = int( floor( ppy ) ); // center pixel

        //std::cout << " px " << px << " py " << py << std::endl;

        int indexPixel = ( py * _width + px ) * _samplePerPixel;
        float r = _images[ index ][ indexPixel ];
        float g = _images[ index ][ indexPixel + 1 ];
        float b = _images[ index ][ indexPixel + 2 ];
        color[0] = r;
        color[1] = g;
        color[2] = b;
        //std::cout << "face " << index << " color " << r << " " << g << " " << b << std::endl;
    }


    void loadEnvFace(TIFF* tif, int face);

    bool loadCubemap(const std::string& name) {

        TIFF *tif;
        tif = TIFFOpen(name.c_str(), "r");
        if (!tif)
            return false;


        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,  &_bitsPerPixel);
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,     &_width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH,    &_height);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &_samplePerPixel);

        std::cout << "reading cubemap environment  6 x " << _width << " x " << _height << " x " << _samplePerPixel << " - " << _bitsPerPixel << " bits" << std::endl;
        for ( int face = 0; face < 6; ++face) {
            loadEnvFace(tif, face);
        }

        TIFFClose(tif);

        return true;
    }

    void computeSpecularAtLevel( float roughness, const CubemapLevel& cubemap) {

        iterateOnFace(0, roughness, cubemap);
        iterateOnFace(1, roughness, cubemap);
        iterateOnFace(2, roughness, cubemap);
        iterateOnFace(3, roughness, cubemap);
        iterateOnFace(4, roughness, cubemap);
        iterateOnFace(5, roughness, cubemap);
    }

    void computeSpecularIrradiance( const std::string& output, int startSize = 0 ) {

        int computeStartSize = startSize;
        if (!computeStartSize)
            computeStartSize = _width;

        int nbMipmap = log2(computeStartSize);
        std::cout << nbMipmap << " mipmap levels will be generated from " << computeStartSize << " x " << computeStartSize << std::endl;

        float start = 0.0;
        float stop = 1.0;

        float step = (stop-start)*1.0/float(nbMipmap);

        for ( int i = 0; i < nbMipmap; i++ ) {
            CubemapLevel cubemap;
            float roughness = step * i;
            uint16 size = pow(2, nbMipmap-i );
            cubemap.init( size, size, _samplePerPixel, _bitsPerPixel);

            std::stringstream ss;
            ss << output << "_" << size << "_" << roughness << ".tif";

            std::cout << "compute specular with roughness " << roughness << " 6 x " << size << " x " << size << " to " << ss.str() << std::endl;
            cubemap.computeSpecularAtLevel( step * i, *this);
            cubemap.write( ss.str() );
        }

    }

};


void CubemapLevel::loadEnvFace(TIFF* tif, int face)
{
    if ( ! TIFFSetDirectory(tif, face) )
        return;

    /* Confirm the parameters of the current face. */
    uint32 w, h;
    uint16 b, c;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,      &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH,     &h);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &c);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,   &b);

    if (! (w == _width && h == _height && b == _bitsPerPixel && c == _samplePerPixel)) {
        std::cerr << "can't read face " << face << std::endl;
        return;
    }

    _images[face] = new float[_width*_height*_samplePerPixel];

    /* Allocate a scanline buffer. */
    std::vector<float> s;
    s.resize(TIFFScanlineSize(tif));

    /* Iterate over all pixels of the current cube face. */
    for (uint32 i = 0; i < _height; ++i) {

        if (TIFFReadScanline(tif, &s.front(), i, 0) == 1) {

            for (uint32 j = 0; j < _width; ++j) {

                for (int k =0; k < _samplePerPixel; k++) {
                    float p = s[ j * c + k ];
                    _images[face][(i*_width + j)*_samplePerPixel + k ] = p;
                }
            }
        }
    }
}


/*----------------------------------------------------------------------------*/

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-s size] in.tif out.tif" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{

    CubemapLevel image;
    int size = 0;
    int c;

    while ((c = getopt(argc, argv, "s:")) != -1)
        switch (c)
        {
        case 's': size = atof(optarg);       break;

        default: return usage(argv[0]);
        }

    std::string input, output;
    if ( optind < argc-1 ) {
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );
    } else {
        return usage( argv[0] );
    }

    image.loadCubemap(input);
    image.computeSpecularIrradiance( output, size );

    return 0;
}
