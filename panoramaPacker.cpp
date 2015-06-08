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

struct PanoramaRGBA8 {

    int _size; // width
    uint8_t* _image;

    void init(int size) {
        _size = size;
        _image = new uint8_t[size*size*4];
    }

    void pack( const std::string& file) {

        FILE* output = fopen( file.c_str(), "wb");

        if (writeByChannel) {
            // write by channel
            for ( int c = 0; c < 4; c++ )
                for ( int b = 0; b < _size*_size; b++ ) {
                    uint8_t* p = &(_image[b*4]);
                    fwrite( &p[c], 1, 1 , output );
                }
        } else {

            fwrite( _image, _size*_size*4, 1 , output );
        }

    }
};



struct PanoramaRGB32 {

    int _size; // width
    float* _image;

    void init(int size) {
        _size = size;
        _image = new float[size*size*3];
    }

    void pack( const std::string& file) {

        FILE* output = fopen( file.c_str(), "wb");

        if (writeByChannel) {
            // write by channel
            for ( int c = 0; c < 3; c++ )
                for ( int b = 0; b < _size*_size; b++ ) {
                    float* p = &(_image[b*3]);
                    fwrite( &p[c], 4, 1 , output );
                }
        } else {

            fwrite( _image, _size*_size*3*4, 1 , output );
        }

    }
};


class Packer
{

public:

    PanoramaRGBA8 _RGBM;
    PanoramaRGBA8 _RGBE;
    PanoramaRGBA8 _LUV;
    PanoramaRGB32 _float;
    std::string _filePattern;
    std::string _outputDirectory;

    int _maxLevel;

    Packer(const std::string& filenamePattern, int level, const std::string& outputDirectory ) {
        _filePattern = filenamePattern;
        _maxLevel = level;
        _outputDirectory = outputDirectory;
    }

    ImageBuf* mipmap() {

        int size = int( pow(2,_maxLevel) );

        ImageSpec specMip( size, size, 3, TypeDesc::FLOAT );
        _float.init( size );
        float* data = _float._image;
        ImageBuf* imageMip = new ImageBuf( specMip , data );

        uint offset = size/2;

        char str[256];

        for ( int level = 0 ; level < _maxLevel + 1; level++) {
            int size = int( pow(2,_maxLevel-level) );

            std::cout << "packing level " << level << " size " << size << std::endl;

            int strSize = snprintf( str, 255, _filePattern.c_str(), level );
            str[strSize+1] = 0;

            ImageBuf src ( str );

            if ( !src.read() || size <= 4 ) {
                std::cout << "skipping file " << str << std::endl;
                continue;
            }

#if 1
            int heightSizeWithoutBorder = size/2 - 2;
            int widthSizeWithoutBorder = size - 2;
            ImageSpec specResize( widthSizeWithoutBorder, heightSizeWithoutBorder, 3, TypeDesc::FLOAT );
            ImageBuf imageResized = ImageBuf( specResize );
            ImageBufAlgo::resize( imageResized, src );

            // handle top / bottom lines
            ImageBufAlgo::paste (*imageMip, 1, offset, 0, 0, imageResized, ROI(0, widthSizeWithoutBorder, 0,1) );
            ImageBufAlgo::paste (*imageMip, 1, offset + size/2 -1 , 0, 0, imageResized, ROI(0, widthSizeWithoutBorder, heightSizeWithoutBorder - 1 , heightSizeWithoutBorder) );


            // left / right
            ImageBufAlgo::paste (*imageMip, 0, offset + 1, 0, 0, imageResized, ROI(widthSizeWithoutBorder-1, widthSizeWithoutBorder, 0,heightSizeWithoutBorder));
            ImageBufAlgo::paste (*imageMip, size - 1, offset + 1, 0, 0, imageResized, ROI(0, 1, 0,heightSizeWithoutBorder) );

            // corner
            Vec3f pixelTmp;
            Vec3f pixel;

            // top
            imageResized.getpixel(widthSizeWithoutBorder - 1, 0, 0, &pixel[0] );
            imageResized.getpixel(0, 0, 0, &pixelTmp[0] );
            pixel = (pixelTmp + pixel) * 0.5;

            // top left
            imageMip->setpixel(0, offset, 0, &pixel[0] );
            // top right
            imageMip->setpixel(size - 1, offset, 0, &pixel[0]);


            // bottom
            imageResized.getpixel(widthSizeWithoutBorder - 1, heightSizeWithoutBorder-1, 0, &pixel[0] );
            imageResized.getpixel(0, heightSizeWithoutBorder-1, 0, &pixelTmp[0] );
            pixel = (pixelTmp + pixel) * 0.5;

            // bottom rigth
            imageMip->setpixel(size - 1, offset + size/2 - 1, 0, &pixel[0]);
            // bottom left
            imageMip->setpixel(0, offset + size/2 - 1, 0, &pixel[0]);


            // image resized
            ImageBufAlgo::paste (*imageMip, 1, offset + 1 , 0, 0, imageResized);

#else
            ImageBufAlgo::paste (*imageMip, 0, offset , 0, 0, src);
#endif
            offset = offset / 2;
        }


        _float.pack(_outputDirectory + "_float.bin");

        imageMip->write("/tmp/debug_panorama_prefilter.tif");

        return imageMip;
    }

    void pack( ImageBuf* image ) {

        int size = image->spec().width;

        _RGBE.init(size);
        _RGBM.init(size);
        _LUV.init(size);

        ImageBuf::Iterator<float, float> iteratorSrc(*image, 0, size, 0, size);


        ImageSpec specOutRGBE(size, size, 4, TypeDesc::UINT8 );
        specOutRGBE.attribute("oiio:UnassociatedAlpha", 1);

        ImageSpec specOutRGBM(size, size, 4, TypeDesc::UINT8 );
        specOutRGBM.attribute("oiio:UnassociatedAlpha", 1);

        ImageSpec specOutLUV(size, size, 4, TypeDesc::UINT8 );
        specOutLUV.attribute("oiio:UnassociatedAlpha", 1);

        ImageBuf dstRGBE(specOutRGBE, _RGBE._image);
        ImageBuf dstRGBM(specOutRGBM, _RGBM._image);
        ImageBuf dstLUV(specOutLUV, _LUV._image);
        ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstRGBE(dstRGBE, 0, size, 0, size);
        ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstRGBM(dstRGBM, 0, size, 0, size);
        ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstLUV(dstLUV, 0, size, 0, size);

        float inTmp[3];
        float* in;
        for (;iteratorDstRGBE.valid();
             iteratorSrc++,
                 iteratorDstRGBE++,
                 iteratorDstRGBM++,
                 iteratorDstLUV++ ) {

            iteratorSrc.pos( iteratorDstRGBE.x(), iteratorDstRGBE.y(), iteratorDstRGBE.z());

            float* inRaw = (float*)iteratorSrc.rawptr();
            uint8_t* outRGBE = (uint8_t*)iteratorDstRGBE.rawptr();
            uint8_t* outRGBM = (uint8_t*)iteratorDstRGBM.rawptr();
            uint8_t* outLUV = (uint8_t*)iteratorDstLUV.rawptr();

            // we assume to have at least 3 channel in inputs, but it could be greyscale
            in = inRaw;

            encodeRGBE(in, outRGBE );
            encodeLUV(in, outLUV );
#if 0
            if ( in[0] != 0 && in[1] != 0 && in[2] != 0) {
                decodeLUV(outLUV, inTmp);

                Vec3f inVec3 = Vec3f(in[0], in[1], in[2]);
                Vec4f result = LogLuvEncode( inVec3 );
                Vec3f result2 = LogLuvDecode(result);
                Vec3f diff = result2 - inVec3;
                std::cout << diff[0] << " " << diff[1] << " " << diff[2] << std::endl;

                unsigned char test2[4];
                Vec3f outVec3;
                encodeLUV( &inVec3[0], test2 );
                decodeLUV( test2, &outVec3[0] );
                Vec3f diff2 = outVec3 - inVec3;
                std::cout << diff2[0] << " " << diff2[1] << " " << diff2[2] << std::endl;


            }
#endif
            encodeRGBM(in, outRGBM );

        }

        _RGBE.pack( _outputDirectory + "_rgbe.bin" );
        _LUV.pack( _outputDirectory + "_luv.bin" );
        _RGBM.pack( _outputDirectory + "_rgbm.bin" );

    }

};


static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-c write by channel] level inputPattern output" << std::endl;
    std::cerr << "eg: " << name << " 5 input_%d.tif /tmp/test/" << std::endl;
    return 1;
}


int main(int argc, char** argv) {

    int level = 0;
    int c;
    writeByChannel = false;
    while ((c = getopt(argc, argv, "c")) != -1)
        switch (c)
        {
        case 'c': writeByChannel = true;     break;

        default: return usage(argv[0]);
        }


    std::string filePattern = argv[1];
    std::string outputDir = argv[3];

    if ( optind < argc-2 ) {

        // generate specular ibl
        filePattern = std::string( argv[optind] );
        level = atof( argv[optind+1] );
        outputDir = std::string( argv[optind+2] );

    }

    Packer packer( filePattern, level,  outputDir );
    packer.pack( packer.mipmap() );
}
