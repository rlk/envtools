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

OIIO_NAMESPACE_USING


// https://gist.github.com/aras-p/1199797
void encodeRGBM( float rgb[3], uint8_t rgbm[4]) {

// in our case,
    const float kRGBMMaxRange = 8.0f;
    const float kOneOverRGBMMaxRange = 1.0f / kRGBMMaxRange;

// encode to RGBM, c = ARGB colors in 0..1 floats

    float r = rgb[0] * kOneOverRGBMMaxRange;
    float g = rgb[1] * kOneOverRGBMMaxRange;
    float b = rgb[2] * kOneOverRGBMMaxRange;

    float a = std::max(std::max(r, g), std::max(b, 1e-6f));
    a = ceilf(a * 255.0f) / 255.0f;

    rgbm[0] = uint8_t( std::min(a, 1.0f) * 255.0 );
    rgbm[1] = uint8_t( std::min(r / a, 1.0f) * 255 );
    rgbm[2] = uint8_t( std::min(g / a, 1.0f) *255 );
    rgbm[3] = uint8_t( std::min(b / a, 1.0f) *255 );
}


void encodeRGBE( float rgb[3], uint8_t rgbe[4]) {

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


// M matrix, for encoding
const static float M[] = {
    0.2209, 0.3390, 0.4184,
    0.1138, 0.6780, 0.7319,
    0.0102, 0.1130, 0.2969 };

Vec4f LogLuvEncode(const Vec3f& vRGB)
{
    Vec4f vResult;
    Vec3f Xp_Y_XYZp;

    Xp_Y_XYZp[0] = vRGB[0] * M[0] + vRGB[1] * M[1] + vRGB[2] * M[2];
    Xp_Y_XYZp[1] = vRGB[0] * M[3] + vRGB[1] * M[4] + vRGB[2] * M[5];
    Xp_Y_XYZp[2] = vRGB[0] * M[6] + vRGB[1] * M[7] + vRGB[2] * M[8];

    Xp_Y_XYZp[0] = std::max(Xp_Y_XYZp[0], 1e-6f );
    Xp_Y_XYZp[1] = std::max(Xp_Y_XYZp[1], 1e-6f );
    Xp_Y_XYZp[2] = std::max(Xp_Y_XYZp[2], 1e-6f );

    Vec2f tmp = Vec2f( Xp_Y_XYZp[0], Xp_Y_XYZp[1] );
    tmp /= Xp_Y_XYZp[2];
    vResult[0] = tmp[0];
    vResult[1] = tmp[1];
    float Le = 2 * log2(Xp_Y_XYZp[1]) + 127;
    vResult[3] = Le - floor( Le );
    vResult[2] = (Le - (floor(vResult[3]*255.0f))/255.0f)/255.0f;
    return vResult;
}

void encodeLUV( float rgb[3], uint8_t luv[4]) {

    Vec4f result = LogLuvEncode( Vec3f(rgb[0], rgb[1], rgb[2] ) );
    luv[0] = uint8_t(result[0]*255);
    luv[1] = uint8_t(result[1]*255);
    luv[2] = uint8_t(result[2]*255);
    luv[3] = uint8_t(result[3]*255);
}

// Inverse M matrix, for decoding
const static float InverseM[] = {
    6.0013,    -2.700,    -1.7995,
    -1.332,    3.1029,    -5.7720,
    .3007,    -1.088,    5.6268 };

Vec3f LogLuvDecode( const Vec4f& vLogLuv)
{
    float Le = vLogLuv[2] * 255 + vLogLuv[3];
    Vec3f Xp_Y_XYZp;
    Xp_Y_XYZp[1] = exp2((Le - 127) / 2);
    Xp_Y_XYZp[2] = Xp_Y_XYZp[1] / vLogLuv[1];
    Xp_Y_XYZp[0] = vLogLuv[0] * Xp_Y_XYZp[2];
    Vec3f vRGB;
    vRGB[0] = Xp_Y_XYZp[0] * InverseM[0] + Xp_Y_XYZp[1] * InverseM[1] + Xp_Y_XYZp[2] * InverseM[2];
    vRGB[1] = Xp_Y_XYZp[0] * InverseM[3] + Xp_Y_XYZp[1] * InverseM[4] + Xp_Y_XYZp[2] * InverseM[5];
    vRGB[2] = Xp_Y_XYZp[0] * InverseM[6] + Xp_Y_XYZp[1] * InverseM[7] + Xp_Y_XYZp[2] * InverseM[8];
    return vRGB.max(0.0);
}
void decodeLUV( uint8_t luv[4], float rgb[3]) {

    Vec3f result = LogLuvDecode( Vec4f(luv[0]*1.0/255.0, luv[1]*1.0/255.0, luv[2]*1.0/255.0, luv[3]*1.0/255.0 ) );
    rgb[0] = result[0];
    rgb[1] = result[1];
    rgb[2] = result[2];
}



struct PanoramaRGBA8 {

    int _size; // width
    uint8_t* _image;

    void init(int size) {
        _size = size;
        _image = new uint8_t[size*size/2*4];
    }

    void pack( FILE* output) {
        fwrite( _image, _size*_size/2*4, 1 , output );
    }

};


class Packer
{

public:
    typedef std::map<int, PanoramaRGBA8 > MapPanorama8;
    MapPanorama8 _RGBM;
    MapPanorama8 _RGBE;
    MapPanorama8 _LUV;
    std::vector<int> _keys;
    std::string _filePattern;
    std::string _outputDirectory;

    int _maxLevel;

    Packer(const std::string& filenamePattern, int level, const std::string& outputDirectory ) {
        _filePattern = filenamePattern;
        _maxLevel = level;
        _outputDirectory = outputDirectory;
    }

    void write(MapPanorama8& panorama, const std::string& file) {

        uint size = panorama[0]._size;
        uint8_t* buffer = new uint8_t[size*size*4];
        ImageSpec spec(size, size, 4, TypeDesc::UINT8 );
        ImageBuf out( spec, buffer );

        uint offset = 0;

        for ( int i = 0; i < _keys.size(); i++ ) {
            int key = _keys[i];
            uint width = panorama[key]._size;
            uint height = width/2;
            ImageSpec specIn(width, height, 4, TypeDesc::UINT8 );
            ImageBuf src(specIn, panorama[key]._image );

            ImageBufAlgo::paste (out, 0, offset , 0, 0, src);

            offset += height;
        }

        FILE* output = fopen( file.c_str(), "wb");
        fwrite( buffer, size*size*4, 1 , output );
        delete [] buffer;
    }


    void pack() {

        char str[256];

        for ( int level = 0 ; level < _maxLevel + 1; level++) {
            int size = int( pow(2,_maxLevel-level) );
            _keys.push_back( size );

            _RGBE[size].init(size);
            _RGBM[size].init(size);
            _LUV[size].init(size);

            std::cout << "packing level " << level << " size " << size << std::endl;

            int strSize = snprintf( str, 255, _filePattern.c_str(), level );
            str[strSize+1] = 0;

            ImageBuf src ( str );
            src.read();

            const ImageSpec& specIn( src.spec() );

            ImageSpec specOutRGBE(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
            specOutRGBE.attribute("oiio:UnassociatedAlpha", 1);

            ImageSpec specOutRGBM(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
            specOutRGBM.attribute("oiio:UnassociatedAlpha", 1);

            ImageSpec specOutLUV(specIn.width, specIn.height, 4, TypeDesc::UINT8 );
            specOutLUV.attribute("oiio:UnassociatedAlpha", 1);

            ImageBuf dstRGBE(specOutRGBE, _RGBE[size]._image);
            ImageBuf dstRGBM(specOutRGBM, _RGBM[size]._image);
            ImageBuf dstLUV(specOutLUV, _LUV[size]._image);

            int width = specIn.width,
                height = specIn.height;

            ImageBuf::Iterator<float, float> iteratorSrc(src, 0, width, 0, height);

            ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstRGBE(dstRGBE, 0, width, 0, height);
            ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstRGBM(dstRGBM, 0, width, 0, height);
            ImageBuf::Iterator<uint8_t, uint8_t> iteratorDstLUV(dstLUV, 0, width, 0, height);

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

            }
        }

        write( _RGBE, _outputDirectory + "_rgbe.bin" );
        write( _RGBM, _outputDirectory + "_rgbm.bin" );
        write( _LUV, _outputDirectory + "_luv.bin" );
    }
};


int main(int argc, char** argv) {

    std::string filePattern = argv[1];
    int level = atof(argv[2]);
    std::string outputDir = argv[3];

    Packer packer( filePattern, level,  outputDir );
    packer.pack();
}
