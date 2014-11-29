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

OIIO_NAMESPACE_USING


void encode( float rgb[3], uint8_t rgbe[4]) {

    float maxRGB = std::max( rgb[0], std::max( rgb[1], rgb[2] ) );

    if(maxRGB < 1e-32) {
        rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
    } else {
        int e;
        float v = frexp(maxRGB, &e) * 256.0 / maxRGB;
        rgbe[0] = (unsigned char)(rgb[0] * v);
        rgbe[1] = (unsigned char)(rgb[1] * v);
        rgbe[2] = (unsigned char)(rgb[2] * v);
        rgbe[3] = (unsigned char)(e + 128);
    }
}



struct CubemapRGBE {

    int _size;
    uint8_t* _images[6];

    void init(int size) {
        _size = size;
        for ( int i = 0; i<6; i++)
            _images[i] = new uint8_t[size*size*4];
    }

    void pack( FILE* output) {
        for ( int i = 0; i < 6; i++ )
            fwrite( _images[i], _size*_size*4, 1 , output );
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
        for ( int i = 0; i < 6; i++ ) {
            fwrite( _images[i], _size*_size*4*3, 1 , output );
        }
    }

};



class Packer
{

public:

    std::map<int, CubemapRGBE > _cubemaps;
    std::map<int, CubemapFloat > _cubemapsFloat;
    std::vector<int> _keys;
    std::string _filePattern;
    std::string _outputDirectory;

    int _maxLevel;

    Packer(const std::string& filenamePattern, int level, const std::string& outputDirectory ) {
        _filePattern = filenamePattern;
        _maxLevel = level;
        _outputDirectory = outputDirectory;
    }

    void pack() {

        char str[256];

        for ( int level = 0 ; level < _maxLevel + 1; level++) {
            int size = int( pow(2,_maxLevel-level) );
            _keys.push_back( size );
            _cubemaps[size].init(size);
            _cubemapsFloat[size].init(size);

            std::cout << "processing size level " << level << " size " << size << std::endl;

            for ( int i = 0 ; i < 6; i++ ) {

                int strSize = snprintf( str, 255, _filePattern.c_str(), level, i );
                str[strSize+1] = 0;

                ImageBuf src(str);
                src.read();
                ImageSpec specIn = src.spec();

                std::cout << "processing " << str << " size " << specIn.width << "x" << specIn.height << std::endl;

                ImageSpec specOut(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
                specOut.attribute("oiio:UnassociatedAlpha", 1);
                ImageBuf dst(specOut, _cubemaps[size]._images[i]);


                ImageSpec specOutFloat;
                specOutFloat.width=specIn.width;
                specOutFloat.height=specIn.height;
                specOutFloat.full_y=0;
                specOutFloat.full_x=0;
                specOutFloat.full_width = specIn.width;
                specOutFloat.full_height = specIn.width;
                specOutFloat.nchannels = specIn.nchannels;
                specOutFloat.tile_height = specIn.height;
                specOutFloat.tile_width = specIn.width;
                specOutFloat.format = TypeDesc::FLOAT;
                specOutFloat.attribute("oiio:ColorSpace", "Linear");
                ImageBuf dstFloat("/tmp/test_super_debug.tif", specOutFloat, _cubemapsFloat[size]._images[i]);


                int width = specIn.width,
                    height = specIn.height;

                ImageBuf::Iterator<float, float> iteratorSrc(src, 0, width, 0, height);
                ImageBuf::Iterator<uint8_t, uint8_t> iteratorDst(dst, 0, width, 0, height);
                ImageBuf::Iterator<float, float> iteratorDstFloat(dstFloat, 0, width, 0, height);

                float result[3];
                float inTmp[3];
                float* in;
                float biggest = 0.0;
                for (; iteratorDst.valid(); iteratorDst++, iteratorSrc++, iteratorDstFloat++) {
                    iteratorSrc.pos( iteratorDst.x(), iteratorDst.y(), iteratorDst.z());

                    float* inRaw = (float*)iteratorSrc.rawptr();
                    uint8_t* out = (uint8_t*)iteratorDst.rawptr();
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

                    encode(in, out );

                    outFloat[0] = in[0];
                    outFloat[1] = in[1];
                    outFloat[2] = in[2];

                }

                dstFloat.save();

                ImageSpec specOutFloat2;
                specOutFloat2.width=specIn.width;
                specOutFloat2.height=specIn.height;
                specOutFloat2.full_y=0;
                specOutFloat2.full_x=0;
                specOutFloat2.full_width = specIn.width;
                specOutFloat2.full_height = specIn.width;
                specOutFloat2.nchannels = specIn.nchannels;
                specOutFloat2.tile_height = specIn.height;
                specOutFloat2.tile_width = specIn.width;
                specOutFloat2.format = TypeDesc::FLOAT;
                specOutFloat2.attribute("oiio:ColorSpace", "Linear");
                ImageBuf dstFloat2( "/tmp/test_super2_debug.tif", specOutFloat2, _cubemapsFloat[size]._images[i]);
                dstFloat2.save();
            }
        }

        FILE* output = fopen( (_outputDirectory + "/cubemap.bin").c_str(), "wb");
        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            _cubemaps[key].pack(output);
        }

        FILE* outputFloat = fopen( (_outputDirectory + "/cubemap_float.bin").c_str() , "wb");
        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            _cubemapsFloat[key].pack(outputFloat);
        }
    }
};


int main(int argc, char** argv) {

    std::string filePattern = argv[1];
    int level = atof(argv[2]);
    std::string outputDir = argv[3];

    Packer packer( filePattern, level,  outputDir );
    packer.pack();
}
