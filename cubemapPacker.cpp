#include <iostream>
#include <getopt.h>
#include <cstdlib>
#include <sstream>
#include <unistd.h>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filter.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include "Math"
#include "Cubemap"
#include "Color"

OIIO_NAMESPACE_USING


bool writeByChannel = false;

struct CubemapRGBA8 {

    int _size;
    uint8_t* _images[6];

    void init(int size) {
        _size = size;
        for ( int i = 0; i<6; i++)
            _images[i] = new uint8_t[size*size*4];
    }

    void pack( FILE* output) {

        if (writeByChannel) {
            // write by channel
            for ( int i = 0; i < 6; i++ )
                for ( int c = 0; c < 4; c++ )
                    for ( int b = 0; b < _size*_size; b++ ) {
                        uint8_t* p = &(_images[i][b*4]);
                        fwrite( &p[c], 1, 1 , output );
                    }
        } else {

            for ( int i = 0; i < 6; i++ )
                fwrite( _images[i], _size*_size*4, 1 , output );
        }

    }

};


struct CubemapFloat {

    int _size;
    float* _images[6];

    void init(int size) {
        _size = size;
        for ( int i = 0; i<6; i++)
            _images[i] = new float[size*size*3];
    }

    void pack( FILE* output) {

        if (writeByChannel) {

            for ( int i = 0; i < 6; i++ )
                for ( int c = 0; c < 3; c++ )
                    for ( int b = 0; b < _size*_size; b++ ) {
                        float* p = &(_images[i][b*3]);
                        fwrite( &p[c], 4, 1 , output );
                    }
        } else {
            for ( int i = 0; i < 6; i++ ) {
                fwrite( _images[i], _size*_size*4*3, 1 , output );
            }
        }
    }

};



class Packer
{

public:

    std::map<int, CubemapRGBA8 > _cubemapsRGBM;
    std::map<int, CubemapRGBA8 > _cubemapsRGBE;
    std::map<int, CubemapRGBA8 > _cubemapsLUV;
    std::map<int, CubemapFloat > _cubemapsFloat;
    std::vector<int> _keys;
    std::string _input;
    std::string _outputDirectory;

    int _maxLevel;

    Packer(const std::string& input, int level, const std::string& outputDirectory ) {
        _input = input;
        _maxLevel = level;
        _outputDirectory = outputDirectory;
    }

    bool processCubemap( uint size, const std::string& name ) {

        Cubemap cm;
        bool loaded = cm.load(name);

        if (!size)
            size = cm.getSize();

        _keys.push_back( size );

        _cubemapsRGBE[size].init(size);
        _cubemapsRGBM[size].init(size);
        _cubemapsLUV[size].init(size);
        _cubemapsFloat[size].init(size);

        if ( !loaded ) {
            return false;
        }


        uint cubemapSize = cm.getSize();
        for ( int i = 0 ; i < 6; i++ ) {

            ImageSpec specIn(cubemapSize, cubemapSize, cm.getSamplePerPixel(), TypeDesc::FLOAT);
            ImageBuf src(specIn, cm.getImages().imageFace(i));

            //std::cout << "processing " << str << " size " << specIn.width << "x" << specIn.height << std::endl;

            ImageSpec specOutRGBE(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
            specOutRGBE.attribute("oiio:UnassociatedAlpha", 1);

            ImageSpec specOutRGBM(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
            specOutRGBM.attribute("oiio:UnassociatedAlpha", 1);

            ImageSpec specOutLUV(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
            specOutLUV.attribute("oiio:UnassociatedAlpha", 1);

            ImageBuf dstRGBE(specOutRGBE, _cubemapsRGBE[size]._images[i]);
            ImageBuf dstRGBM(specOutRGBM, _cubemapsRGBM[size]._images[i]);
            ImageBuf dstLUV(specOutLUV, _cubemapsLUV[size]._images[i]);


            ImageSpec specOutFloat(specIn.width, specIn.height, 3, TypeDesc::FLOAT );
            ImageBuf dstFloat("/tmp/test_super_debug.tif", specOutFloat, _cubemapsFloat[size]._images[i]);


            int width = specIn.width,
                height = specIn.height;

            ImageBuf::Iterator<float, float> iteratorSrc(src, 0, width, 0, height);

            ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstRGBE(dstRGBE, 0, width, 0, height);
            ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstRGBM(dstRGBM, 0, width, 0, height);
            ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstLUV(dstLUV, 0, width, 0, height);
            ImageBuf::Iterator<float, float> iteratorDstFloat(dstFloat, 0, width, 0, height);

            float result[3];
            float inTmp[3];
            float* in;
            float biggest = 0.0;
            for (;iteratorDstRGBE.valid();
                 iteratorSrc++,
                     iteratorDstRGBE++,
                     iteratorDstRGBM++,
                     iteratorDstLUV++,
                     iteratorDstFloat++) {

                iteratorSrc.pos( iteratorDstRGBE.x(), iteratorDstRGBE.y(), iteratorDstRGBE.z());

                float* inRaw = (float*)iteratorSrc.rawptr();
                uint8_t* outRGBE = (uint8_t*)iteratorDstRGBE.rawptr();
                uint8_t* outRGBM = (uint8_t*)iteratorDstRGBM.rawptr();
                uint8_t* outLUV = (uint8_t*)iteratorDstLUV.rawptr();
                float* outFloat = (float*)iteratorDstFloat.rawptr();

                // we assume to have at least 3 channel in inputs, but it could be greyscale
                if ( specIn.nchannels < 3 ) {
                    inTmp[0] = inRaw[0];
                    inTmp[1] = inRaw[0];
                    inTmp[2] = inRaw[0];
                    in = inTmp;
                } else {
                    in = inRaw;
                }

                encodeRGBE(in, outRGBE );
                encodeLUV(in, outLUV );
                encodeRGBM(in, outRGBM );

                outFloat[0] = in[0];
                outFloat[1] = in[1];
                outFloat[2] = in[2];

            }
        }
        return true;
    }

    void pack() {

        char str[256];

        // when using pattern options
        // pack all miplevel
        if ( _maxLevel > 0 ) {

            for ( int level = 0 ; level < _maxLevel + 1; level++) {
                int size = int( pow(2,_maxLevel-level) );

                std::cout << "packing level " << level << " size " << size << std::endl;
                int strSize = snprintf( str, 255, _input.c_str(), level );
                str[strSize+1] = 0;

                bool result = processCubemap( size, str );
                if (!result)
                    std::cout << "can't read cubemap " << str << " for size " << size << ", skipped" << std::endl;
            }

        } else {

            // when using pattern options
            bool result = processCubemap( 0, _input );
            if (!result)
                std::cout << "error can't read file " << _input << std::endl;

        }

        FILE* outputRGBE = fopen( (_outputDirectory + "_rgbe.bin").c_str(), "wb");
        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            _cubemapsRGBE[key].pack(outputRGBE);
        }

        FILE* outputRGBM = fopen( (_outputDirectory + "_rgbm.bin").c_str(), "wb");
        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            _cubemapsRGBM[key].pack(outputRGBM);
        }

        FILE* outputLUV = fopen( (_outputDirectory + "_luv.bin").c_str(), "wb");
        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            _cubemapsLUV[key].pack(outputLUV);
        }

        FILE* outputFloat = fopen( (_outputDirectory + "_float.bin").c_str() , "wb");
        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            _cubemapsFloat[key].pack(outputFloat);
        }
    }

};

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-c write by channel] [-p toogle pattern] [-n nb level] input.tif outputdirectory" << std::endl;
    std::cerr << "eg: " << name << " -p -n 5 input_%d.tif /tmp/test/" << std::endl;
    std::cerr << "eg: " << name << "input.tif /tmp/test/" << std::endl;
    return 1;
}

int main(int argc, char** argv) {

    bool pattern = false;
    int nb = 0;
    int c;
    writeByChannel = false;
    while ((c = getopt(argc, argv, "cpn:")) != -1)
        switch (c)
        {
        case 'c': writeByChannel = true;     break;
        case 'p': pattern = true;     break;
        case 'n': nb = atoi(optarg);  break;

        default: return usage(argv[0]);
        }


    std::string input, output;
    if ( optind < argc-1 ) {

        // generate specular ibl
        input = std::string( argv[optind] );
        output = std::string( argv[optind+1] );

        Packer packer( input, nb,  output );
        packer.pack();

    } else {
        return usage( argv[0] );
    }

    return 0;
}
