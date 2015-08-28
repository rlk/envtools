#include <sys/stat.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "Math"
#include "Cubemap"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filter.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

OIIO_NAMESPACE_USING

void texelCoordToVect(int face, float ui, float vi, int size, float* dirResult, int fixup = 0);
void vectToTexelCoord(const Vec3f& direction, int size, int& faceIndex, float& u, float& v);

Cubemap::Cubemap()
{
    _levels.resize(1);
}


Cubemap::~Cubemap()
{
}


Cubemap::MipLevel::MipLevel()
{
    _size = 0;
    for ( int i = 0; i < 6; i++ ) {
        _images[i] = 0;
    }
}


Cubemap::MipLevel::~MipLevel()
{
    for ( int i = 0; i < 6; i++ ) {
        if ( _images[i] )
            delete [] _images[i];
    }
}


void Cubemap::MipLevel::init( uint size, uint sample )
{
    _size = size;
    _samplePerPixel = sample;
    for ( int i = 0; i < 6; i++ ) {
        if (_images[i])
            delete [] _images[i];
        _images[i] = new float[size*size*sample];
    }
}


void Cubemap::Cubemap::init( int size, int sample )
{
    _levels[0].init( size, sample );
}

void Cubemap::fill( const Vec4f& fillValue )
{
    uint size = getSize();
    uint samplePerPixel = getSamplePerPixel();
    uint totalFloat = size * size * samplePerPixel;;

    for ( int i = 0; i < 6; i++ ) {
        float* data = getImages().imageFace(i);

        if ( samplePerPixel > 3 ) {
            for ( uint j = 0; j < totalFloat; j += samplePerPixel ) {
                data[j]   = fillValue[0];
                data[j+1] = fillValue[1];
                data[j+2] = fillValue[2];
                data[j+3] = fillValue[3];
            }

        } else {

            for ( uint j = 0; j < totalFloat; j += samplePerPixel ) {
                data[j]   = fillValue[0];
                data[j+1] = fillValue[1];
                data[j+2] = fillValue[2];
            }

        }
    }
}



// SH order use for approximation of irradiance cubemap is 5, mean 5*5 equals 25 coefficients
#define MAX_SH_ORDER 5
#define NUM_SH_COEFFICIENT (MAX_SH_ORDER * MAX_SH_ORDER)

// See Peter-Pike Sloan paper for these coefficients
static double SHBandFactor[NUM_SH_COEFFICIENT] = { 1.0,
                                                    2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
                                                    1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0,
                                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // The 4 band will be zeroed
                                                    - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0, - 1.0 / 24.0};

void EvalSHBasis(const float* dir, double* res )
{
    // Can be optimize by precomputing constant.
    static const double SqrtPi = sqrt(PI);

    double xx = dir[0];
    double yy = dir[1];
    double zz = dir[2];

    // x[i] == pow(x, i), etc.
    double x[MAX_SH_ORDER+1], y[MAX_SH_ORDER+1], z[MAX_SH_ORDER+1];
    x[0] = y[0] = z[0] = 1.;
    for (int i = 1; i < MAX_SH_ORDER+1; ++i)
    {
        x[i] = xx * x[i-1];
        y[i] = yy * y[i-1];
        z[i] = zz * z[i-1];
    }

    res[0]  = (1/(2.*SqrtPi));

    res[1]  = -(sqrt(3/PI)*yy)/2.;
    res[2]  = (sqrt(3/PI)*zz)/2.;
    res[3]  = -(sqrt(3/PI)*xx)/2.;

    res[4]  = (sqrt(15/PI)*xx*yy)/2.;
    res[5]  = -(sqrt(15/PI)*yy*zz)/2.;
    res[6]  = (sqrt(5/PI)*(-1 + 3*z[2]))/4.;
    res[7]  = -(sqrt(15/PI)*xx*zz)/2.;
    res[8]  = sqrt(15/PI)*(x[2] - y[2])/4.;

    res[9]  = (sqrt(35/(2.*PI))*(-3*x[2]*yy + y[3]))/4.;
    res[10] = (sqrt(105/PI)*xx*yy*zz)/2.;
    res[11] = -(sqrt(21/(2.*PI))*yy*(-1 + 5*z[2]))/4.;
    res[12] = (sqrt(7/PI)*zz*(-3 + 5*z[2]))/4.;
    res[13] = -(sqrt(21/(2.*PI))*xx*(-1 + 5*z[2]))/4.;
    res[14] = (sqrt(105/PI)*(x[2] - y[2])*zz)/4.;
    res[15] = -(sqrt(35/(2.*PI))*(x[3] - 3*xx*y[2]))/4.;

    res[16] = (3*sqrt(35/PI)*xx*yy*(x[2] - y[2]))/4.;
    res[17] = (-3*sqrt(35/(2.*PI))*(3*x[2]*yy - y[3])*zz)/4.;
    res[18] = (3*sqrt(5/PI)*xx*yy*(-1 + 7*z[2]))/4.;
    res[19] = (-3*sqrt(5/(2.*PI))*yy*zz*(-3 + 7*z[2]))/4.;
    res[20] = (3*(3 - 30*z[2] + 35*z[4]))/(16.*SqrtPi);
    res[21] = (-3*sqrt(5/(2.*PI))*xx*zz*(-3 + 7*z[2]))/4.;
    res[22] = (3*sqrt(5/PI)*(x[2] - y[2])*(-1 + 7*z[2]))/8.;
    res[23] = (-3*sqrt(35/(2.*PI))*(x[3] - 3*xx*y[2])*zz)/4.;
    res[24] = (3*sqrt(35/PI)*(x[4] - 6*x[2]*y[2] + y[4]))/16.;
}



void Cubemap::getSample(const Vec3f& direction, Vec3f& color ) const
{
    _levels[0].getSample(direction, color);
}

/** Original code from Ignacio Castaño
* This formula is from Manne Öhrström's thesis.
* Take two coordiantes in the range [-1, 1] that define a portion of a
* cube face and return the area of the projection of that portion on the
* surface of the sphere.
**/

static float AreaElement( float x, float y )
{
    return atan2(x * y, sqrt(x * x + y * y + 1));
}

float Cubemap::MipLevel::texelCoordSolidAngle(uint faceIdx, float aU, float aV) const
{
    // transform from [0..res - 1] to [- (1 - 1 / res) .. (1 - 1 / res)]
    // (+ 0.5f is for texel center addressing)
    float U = (2.0f * ((float)aU + 0.5f) / (float)_size ) - 1.0f;
    float V = (2.0f * ((float)aV + 0.5f) / (float)_size ) - 1.0f;

    // Shift from a demi texel, mean 1.0f / a_Size with U and V in [-1..1]
    float InvResolution = 1.0f / _size;

    // U and V are the -1..1 texture coordinate on the current face.
    // Get projected area for this texel
    float x0 = U - InvResolution;
    float y0 = V - InvResolution;
    float x1 = U + InvResolution;
    float y1 = V + InvResolution;
    float SolidAngle = AreaElement(x0, y0) - AreaElement(x0, y1) - AreaElement(x1, y0) + AreaElement(x1, y1);

    return SolidAngle;
}

float Cubemap::texelCoordSolidAngle(uint faceIdx, float aU, float aV) const
{
    return _levels[0].texelCoordSolidAngle(faceIdx, aU, aV);
}


void Cubemap::buildNormalizerSolidAngleCubemap(uint size, int fixup)
{
    _levels[0].buildNormalizerSolidAngleCubemap(size, fixup);
}

void Cubemap::MipLevel::buildNormalizerSolidAngleCubemap(uint size, int fixup)
{

    init(size, 4);
    uint iCubeFace, u, v;

    //iterate over cube faces
    for (iCubeFace=0; iCubeFace<6; iCubeFace++) {

        //fast texture walk, build normalizer cube map
        float *texelPtr = _images[iCubeFace];

        for(v=0; v < size; v++) {

            for(u=0; u < size; u++) {

                texelCoordToVect(iCubeFace, (float)u, (float)v, size, texelPtr, fixup);
                *(texelPtr + 3) = texelCoordSolidAngle(iCubeFace, (float)u, (float)v);
                texelPtr += _samplePerPixel;

            }
        }
    }
}


Cubemap* Cubemap::shFilterCubeMap(bool useSolidAngleWeighting, int fixup, int outputCubemapSize)
{
    Cubemap* srcCubemap = this;
    Cubemap* dstCubemap = new Cubemap();
    dstCubemap->init(outputCubemapSize, 3 );

    int srcSize = srcCubemap->getSize();
    int dstSize = dstCubemap->getSize();

    //pointers used to walk across the image surface
    float *normCubeRowStartPtr;
    float *srcCubeRowStartPtr;
    float *dstCubeRowStartPtr;
    float *texelVect;

    const int srcCubeMapNumChannels	= srcCubemap->getSamplePerPixel();
    const int dstCubeMapNumChannels	= dstCubemap->getSamplePerPixel(); //DstCubeImage[0].m_NumChannels;

    //First step - Generate SH coefficient for the diffuse convolution

    //Regenerate normalization cubemap
    //clear pre-existing normalizer cube map
    // for(int iCubeFace=0; iCubeFace<6; iCubeFace++)
    // {
    // 	m_NormCubeMap[iCubeFace].Clear();
    // }

    Cubemap normCubemap = Cubemap();

    //Normalized vectors per cubeface and per-texel solid angle
    normCubemap.buildNormalizerSolidAngleCubemap(srcCubemap->getSize(), fixup);

    const int normCubeMapNumChannels = normCubemap.getSamplePerPixel(); // This need to be init here after the generation of m_NormCubeMap

    //This is a custom implementation of D3DXSHProjectCubeMap to avoid to deal with LPDIRECT3DSURFACE9 pointer
    //Use Sh order 2 for a total of 9 coefficient as describe in http://www.cs.berkeley.edu/~ravir/papers/envmap/
    //accumulators are 64-bit floats in order to have the precision needed
    //over a summation of a large number of pixels
    double SHr[NUM_SH_COEFFICIENT];
    double SHg[NUM_SH_COEFFICIENT];
    double SHb[NUM_SH_COEFFICIENT];
    double SHdir[NUM_SH_COEFFICIENT];

    memset(SHr, 0, NUM_SH_COEFFICIENT * sizeof(double));
    memset(SHg, 0, NUM_SH_COEFFICIENT * sizeof(double));
    memset(SHb, 0, NUM_SH_COEFFICIENT * sizeof(double));
    memset(SHdir, 0, NUM_SH_COEFFICIENT * sizeof(double));

    double weightAccum = 0.0;
    double weight = 0.0;

    for (int iFaceIdx = 0; iFaceIdx < 6; iFaceIdx++)
    {
        for (int y = 0; y < srcSize; y++)
        {
            normCubeRowStartPtr = &normCubemap.getImages().imageFace(iFaceIdx)[ normCubeMapNumChannels * (y * srcSize)];
            srcCubeRowStartPtr	= &srcCubemap->getImages().imageFace(iFaceIdx)[ srcCubeMapNumChannels * (y * srcSize)];

            for (int x = 0; x < srcSize; x++)
            {
                //pointer to direction and solid angle in cube map associated with texel
                texelVect = &normCubeRowStartPtr[normCubeMapNumChannels * x];

                if( useSolidAngleWeighting )
                {   //solid angle stored in 4th channel of normalizer/solid angle cube map
                    weight = *(texelVect+3);
                }
                else
                {   //all taps equally weighted
                    weight = 1.0;
                }

                EvalSHBasis(texelVect, SHdir);

                // Convert to double
                double R = srcCubeRowStartPtr[(srcCubeMapNumChannels * x) + 0];
                double G = srcCubeRowStartPtr[(srcCubeMapNumChannels * x) + 1];
                double B = srcCubeRowStartPtr[(srcCubeMapNumChannels * x) + 2];

                for (int i = 0; i < NUM_SH_COEFFICIENT; i++)
                {
                    SHr[i] += R * SHdir[i] * weight;
                    SHg[i] += G * SHdir[i] * weight;
                    SHb[i] += B * SHdir[i] * weight;
                }

                weightAccum += weight;
            }
        }
    }

    //Normalization - The sum of solid angle should be equal to the solid angle of the sphere (4 PI), so
    // normalize in order our weightAccum exactly match 4 PI.
    for (int i = 0; i < NUM_SH_COEFFICIENT; ++i)
    {
        SHr[i] *= 4.0 * PI / weightAccum;
        SHg[i] *= 4.0 * PI / weightAccum;
        SHb[i] *= 4.0 * PI / weightAccum;
    }

    //Second step - Generate cubemap from SH coefficient

    // regenerate normalization cubemap for the destination cubemap
    //clear pre-existing normalizer cube map
    // for(int iCubeFace=0; iCubeFace<6; iCubeFace++)
    // {
    //     normCubemap[iCubeFace].Clear();
    // }

    //Normalized vectors per cubeface and per-texel solid angle
    //BuildNormalizerSolidAngleCubemap(DstCubeImage->m_Width, m_NormCubeMap, a_FixupType);
    normCubemap.buildNormalizerSolidAngleCubemap(dstCubemap->getSize(), fixup );


    // dump spherical harmonics coefficient
    // shRGB[I] * BandFactor[I]
    std::cout << "shR: [ " << SHr[0] * SHBandFactor[0];
    for (int i = 1; i < NUM_SH_COEFFICIENT; ++i)
        std::cout << ", " << SHr[i] * SHBandFactor[i];
    std::cout << " ]" << std::endl;

    std::cout << "shG: [ " << SHg[0] * SHBandFactor[0];
    for (int i = 1; i < NUM_SH_COEFFICIENT; ++i)
        std::cout << ", " << SHg[i] * SHBandFactor[i];
    std::cout << " ]" << std::endl;

    std::cout << "shB: [ " << SHb[0] * SHBandFactor[0];
    for (int i = 0; i < NUM_SH_COEFFICIENT; ++i)
        std::cout << ", " << SHb[i] * SHBandFactor[i];
    std::cout << " ]" << std::endl;

    std::cout << std::endl;

    std::cout << "shCoef: [ " << SHr[0] * SHBandFactor[0] << ", " << SHg[0] * SHBandFactor[0] << ", " << SHb[0] * SHBandFactor[0];
    for (int i = 1; i < NUM_SH_COEFFICIENT; ++i) {
        std::cout << ", " << SHr[i] * SHBandFactor[i] << ", " << SHg[i] * SHBandFactor[i] << ", " << SHb[i] * SHBandFactor[i];
    }
    std::cout << " ]" << std::endl;


    for (int iFaceIdx = 0; iFaceIdx < 6; iFaceIdx++)
    {
        for (int y = 0; y < dstSize; y++)
        {
            normCubeRowStartPtr = &normCubemap.getImages().imageFace(iFaceIdx)[normCubeMapNumChannels * (y * dstSize)];
            dstCubeRowStartPtr	= &dstCubemap->getImages().imageFace(iFaceIdx)[dstCubeMapNumChannels * (y * dstSize)];

            for (int x = 0; x < dstSize; x++)
            {
                //pointer to direction and solid angle in cube map associated with texel
                texelVect = &normCubeRowStartPtr[normCubeMapNumChannels * x];

                EvalSHBasis(texelVect, SHdir);

                // get color value
                float R = 0.0f, G = 0.0f, B = 0.0f;

                for (int i = 0; i < NUM_SH_COEFFICIENT; ++i)
                {
                    R += (float)(SHr[i] * SHdir[i] * SHBandFactor[i]);
                    G += (float)(SHg[i] * SHdir[i] * SHBandFactor[i]);
                    B += (float)(SHb[i] * SHdir[i] * SHBandFactor[i]);
                }

                dstCubeRowStartPtr[(dstCubeMapNumChannels * x) + 0] = R;
                dstCubeRowStartPtr[(dstCubeMapNumChannels * x) + 1] = G;
                dstCubeRowStartPtr[(dstCubeMapNumChannels * x) + 2] = B;
                if (dstCubeMapNumChannels > 3)
                {
                    dstCubeRowStartPtr[(dstCubeMapNumChannels * x) + 3] = 1.0f;
                }
            }
        }
    }
    return dstCubemap;
}


void Cubemap::MipLevel::write( const std::string& filename ) const
{
    ImageOutput* out = ImageOutput::create (filename);

    // Use Create mode for the first level.
    ImageOutput::OpenMode appendmode = ImageOutput::Create;

    // Write the individual subimages
    for (int s = 0;  s < 6;  ++s) {
        ImageSpec spec( _size, _size, _samplePerPixel, TypeDesc::FLOAT);
        out->open (filename, spec, appendmode);
        out->write_image (TypeDesc::FLOAT, _images[s]);
        // Use AppendSubimage mode for subsequent levels
        appendmode = ImageOutput::AppendSubimage;
    }
    out->close ();
    delete out;
}

void Cubemap::write( const std::string& filename ) const
{
    _levels[0].write(filename);
}


bool Cubemap::MipLevel::load(const std::string& name)
{
    ImageInput* input = ImageInput::open ( name );
    if ( !input )
        return false;

    for ( int i = 0; i < 6; i++) {
        ImageSpec spec;
        input->seek_subimage(i, 0, spec );

        if ( !getSize() ) {

            if ( spec.nchannels <3 ) {
                std::cout << "error your cubemap should have at least 3 channels" << std::endl;
                return false;
            }
            init( spec.width, spec.nchannels );
        }

        if ( spec.width != spec.height && spec.width != getSize() ) {
            std::cout << "Size of sub image " << i << " is not correct" << std::endl;
            return false;
        }
        input->read_image( TypeDesc::FLOAT, _images[i]);
    }
    input->close();
    delete input;
    return true;
}

bool Cubemap::load(const std::string& filename)
{
    return _levels[0].load( filename );
}


bool fileExist(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

bool Cubemap::loadMipMap(const std::string& filenamePattern)
{
    std::vector<std::string> filenames;
    uint maxMipLevel = 30; // should be really enough
    char str[512];
    for ( uint i = 0; i < maxMipLevel; i++ ) {
        int strSize = snprintf( str, 511, filenamePattern.c_str(), i );
        str[strSize+1] = 0;
        std::string filename = std::string( str );
        if ( fileExist(filename) )
             filenames.push_back( filename );
    }

    uint nbMipLevel = filenames.size();
    uint size = pow(2, nbMipLevel-1 );
    std::cout << "found " << nbMipLevel << " mip level - " <<  size << " x " << size << " cubemap" << std::endl;

    _levels.resize(nbMipLevel);
    for ( uint i = 0 ; i < nbMipLevel; i++ ) {
        _levels[i].load(filenames[i]);
    }

    return true;
}

void Cubemap::computePrefilteredEnvironmentUE4( const std::string& output, int startSize, int endSize, uint nbSamples, const bool fixup ) {

    int computeStartSize = startSize;
    if (!computeStartSize)
        computeStartSize = getSize();

    int totalMipmap = log2(computeStartSize);
    int endMipMap = totalMipmap - log2( endSize );
#if 0
    std::set<double> hamm;
    for ( uint i = 0; i < 120000; i++ ) {
        double v = radicalInverse_VdC(i);
        if ( hamm.find(v) != hamm.end() ) {
            std::cout << "entry " << v << " already in map" << std::endl;
            hamm.insert(v);
        }
    }
#endif

    std::cout << endMipMap + 1 << " mipmap levels will be generated from " << computeStartSize << " x " << computeStartSize << " to " << endSize << " x " << endSize << std::endl;

    float start = 0.0;
    float stop = 1.0;

    float step = (stop-start)*1.0/float(endMipMap);

    for ( int i = 0; i < totalMipmap+1; i++ ) {
        Cubemap cubemap;

        // frostbite, lagarde paper p67
        // http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf
        float r = step * i;
        float roughnessLinear = r; //pow(r,1.5);

        int size = pow(2, totalMipmap-i );
        cubemap.init( size );

        std::stringstream ss;
        ss << output << "_" << i << ".tif";

        // generate debug color cubemap after limit size
        if ( i <= endMipMap ) {
            std::cout << "compute level " << i << " with roughness " << roughnessLinear << " " << size << " x " << size << " to " << ss.str() << std::endl;
            cubemap.computePrefilterCubemapAtLevel( roughnessLinear, *this, nbSamples, fixup);
        } else {
            cubemap.fill(Vec4f(1.0,0.0,1.0,1.0));
        }
        cubemap.write( ss.str().c_str() );
    }
}

void Cubemap::computePrefilterCubemapAtLevel( float roughnessLinear, const Cubemap& inputCubemap, uint nbSamples, bool fixup ) {

    roughnessLinear = clampTo(roughnessLinear, 0.0f, 1.0f);

    if ( roughnessLinear == 0.0 )
        nbSamples = 1;

    precomputedLightInLocalSpace( nbSamples, roughnessLinear, inputCubemap.getSize() );

    iterateOnFace(0, roughnessLinear, inputCubemap, nbSamples, fixup);
    iterateOnFace(1, roughnessLinear, inputCubemap, nbSamples, fixup);
    iterateOnFace(2, roughnessLinear, inputCubemap, nbSamples, fixup);
    iterateOnFace(3, roughnessLinear, inputCubemap, nbSamples, fixup);
    iterateOnFace(4, roughnessLinear, inputCubemap, nbSamples, fixup);
    iterateOnFace(5, roughnessLinear, inputCubemap, nbSamples, fixup);
}


void Cubemap::iterateOnFace( uint face, float roughnessLinear, const Cubemap& cubemap, uint nbSamples, bool fixup, bool backgroundAverage ) {


    // find native resolution to copy pixel
    uint size = getSize();
    uint nativeResolution = 0;
    for ( uint i = 0; i < cubemap._levels.size(); i++) {
        if ( cubemap.getImages(i).getSize() == size ) {
            nativeResolution = i;
            break;
        }
    }
    float* dataFace = getImages().imageFace(face);

    for ( uint j = 0; j < size; j++ ) {
        int lineIndex = j*getSamplePerPixel()*size;

#pragma omp parallel for
        for ( uint i = 0; i < size; i++ ) {

            Vec3f direction, resultColor;
            int index = lineIndex + i*getSamplePerPixel();

            texelCoordToVect( face, float(i), float(j), size, &direction[0], fixup ? 1 : 0 );

            if ( roughnessLinear == 0.0 || nbSamples == 1) { // use a copy from the good mipmap level
                cubemap.getImages(nativeResolution).getSample( direction, resultColor);

            } else {

                if ( backgroundAverage ) {
                    resultColor = cubemap.averageEnvMap( direction, nbSamples );
                } else {
                    resultColor = cubemap.prefilterEnvMapUE4( direction, nbSamples );
                }

            }

            dataFace[ index     ] = resultColor[0];
            dataFace[ index + 1 ] = resultColor[1];
            dataFace[ index + 2 ] = resultColor[2];

        }
    }
}


void Cubemap::computeBackground( const std::string& output, int startSize, uint nbSamples, float radius , const bool fixup ) {

    int computeStartSize = startSize;
    if (!computeStartSize)
        computeStartSize = getSize();

    Cubemap cubemap;

    int size = computeStartSize;
    cubemap.init( size );

    radius = clampTo(radius, 0.0f, 1.0f);

    // http://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size
    // http://www.researchgate.net/post/Calculate_the_Gaussian_filters_sigma_using_the_kernels_size
    // http://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel

    // we are not in pixel but in distance on a circle
    //n /= blurSize;
    float sigma = radius/3.0; // 3*sigma rules
    float sigmaSqr = sigma * sigma;

    precomputeUniformSampleOnCone( nbSamples, radius, sigmaSqr );

    cubemap.iterateOnFace(0, radius, *this, nbSamples, fixup, true);
    cubemap.iterateOnFace(1, radius, *this, nbSamples, fixup, true);
    cubemap.iterateOnFace(2, radius, *this, nbSamples, fixup, true);
    cubemap.iterateOnFace(3, radius, *this, nbSamples, fixup, true);
    cubemap.iterateOnFace(4, radius, *this, nbSamples, fixup, true);
    cubemap.iterateOnFace(5, radius, *this, nbSamples, fixup, true);

    cubemap.write( output.c_str() );
}

Vec3f Cubemap::prefilterEnvMapUE4( const Vec3f& R, const uint numSamples ) const
{

    Vec3f N = R;

    Vec3d prefilteredColor = Vec3d(0,0,0);
    Vec3f color;

    Vec3f UpVector = fabs(N[2]) < 0.999 ? Vec3f(0,0,1) : Vec3f(1,0,0);
    Vec3f TangentX = normalize( cross( UpVector, N ) );
    Vec3f TangentY = normalize( cross( N, TangentX ) );

    bool useLod = _levels.size() > 1;

    // see getPrecomputedLightInLocalSpace in Math
    // and https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
    // for the simplification

    Vec3f LworldSpace;
    if (useLod) {

        // optimized lod version
        for( uint i = 0; i < numSamples; i++ ) {
            // vec4 contains the light vector + miplevel
            const Vec4f& L = getPrecomputedLightInLocalSpace( i );
            float precomputedLod = L[3];
            float NoL = L[2];
            LworldSpace = TangentX * L[0] + TangentY * L[1] + N * L[2];

            getSampleLOD( precomputedLod, LworldSpace, color );

            prefilteredColor += Vec3d( color * NoL );
        }

    } else {

        // no lod version
        for( uint i = 0; i < numSamples; i++ ) {
            // vec4 contains the light vector + miplevel
            const Vec4f& L = getPrecomputedLightInLocalSpace( i );
            float NoL = L[2];
            LworldSpace = TangentX * L[0] + TangentY * L[1] + N * L[2];

            getSample( LworldSpace, color );

            prefilteredColor += Vec3d( color * NoL );
        }
    }

    return prefilteredColor / getPrecomputedLightTotalWeight();
}


// same but do a average to compute the background blur
Vec3f Cubemap::averageEnvMap( const Vec3f& R, const uint numSamples ) const {

    Vec3f N = R;
    Vec3d prefilteredColor = Vec3d(0,0,0);
    Vec3f color,direction;

    Vec3f UpVector = fabs(N[2]) < 0.999 ? Vec3f(0,0,1) : Vec3f(1,0,0);
    Vec3f TangentX = normalize( cross( UpVector, N ) );
    Vec3f TangentY = normalize( cross( N, TangentX ) );

    for( uint i = 0; i < numSamples; i++ ) {

        // vec4 contains direction and weight
        const Vec4f& H =  getUniformSampleOnCone( i );

        // localspace to world space
        direction = TangentX * H[0] + TangentY * H[1] + N * H[2];

        getSample( direction, color );
        prefilteredColor += color * H[3];
    }

    return prefilteredColor / getUniformSampleOnConeWeightSum();
}



void texelCoordToVect(int face, float ui, float vi, int size, float* dirResult, int fixup) {

    float u,v;

    if ( fixup ) {
        // Code from Nvtt : http://code.google.com/p/nvidia-texture-tools/source/browse/trunk/src/nvtt/CubeSurface.cpp

        // transform from [0..res - 1] to [-1 .. 1], match up edges exactly.
        u = (2.0f * ui / (size - 1.0f) ) - 1.0f;
        v = (2.0f * vi / (size - 1.0f) ) - 1.0f;

    } else {

        // center ray on texel center
        // generate a vector for each texel
        u = (2.0f * (ui + 0.5f) / size ) - 1.0f;
        v = (2.0f * (vi + 0.5f) / size ) - 1.0f;

    }

    Vec3f vecX = CubemapFace[face][0] * u;
    Vec3f vecY = CubemapFace[face][1] * v;
    Vec3f vecZ = CubemapFace[face][2];
    Vec3f res = Vec3f( vecX + vecY + vecZ );
    res.normalize();
    dirResult[0] = res[0];
    dirResult[1] = res[1];
    dirResult[2] = res[2];
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

void vectToTexelCoord(const Vec3f& direction, int size, int& faceIndex, float& u, float& v) {

    int bestAxis = 0;
    if ( fabs(direction[1]) > fabs(direction[0]) ) {
        bestAxis = 1;
        if ( fabs(direction[2]) > fabs(direction[1]) )
            bestAxis = 2;
    } else if ( fabs(direction[2]) > fabs(direction[0]) )
        bestAxis = 2;

    // select the index of cubemap face
    faceIndex = bestAxis*2 + ( direction[bestAxis] > 0 ? 0 : 1 );
    float bestAxisValue = direction[bestAxis];
    float denom = fabs( bestAxisValue );

    //float maInv = 1.0/denom;
    Vec3f dir = direction * 1.0/denom;

    float sc = CubemapFace[faceIndex][0] * dir;
    float tc = CubemapFace[faceIndex][1] * dir;
    float ppx = (sc + 1.0) * 0.5 * (size - 1); // width == height
    float ppy = (tc + 1.0) * 0.5 * (size - 1); // width == height

    // u = int( floor( ppx ) ); // center pixel
    // v = int( floor( ppy ) ); // center pixel
    u = ppx;
    v = ppy;

}

void Cubemap::getSampleLOD(float lod, const Vec3f& direction, Vec3f& color ) const {
    float l0 = floor( lod );
    float l1 = ceil( lod );
    float r = lod - l0;

    Vec3f color0,color1;
    _levels[int(l0)].getSample(direction, color0 );
    _levels[int(l1)].getSample(direction, color1 );
    color = lerp( color0, color1, r );
}

void Cubemap::MipLevel::getSample(const Vec3f& direction, Vec3f& color ) const {

    float u,v;
    int faceIndex;

    int size = getSize();
    vectToTexelCoord(direction, size, faceIndex, u,v );


    const float ii = clamp(u - 0.5f, 0.0f, size - 1.0f);
    const float jj = clamp(v - 0.5f, 0.0f, size - 1.0f);

    // const long  i0 = lrintf(ii);
    // const long  j0 = lrintf(jj);

    const long  i0 = lrintf(u);
    const long  j0 = lrintf(v);


    // for ( int i = 0; i < 3; i++ )
    //     color[i] = lerp( lerp( _images[ faceIndex ][ ( j0 * size + i0 ) * getSamplePerPixel() + i  ],
    //                             _images[ faceIndex ][ ( j0 * size + i1 ) * getSamplePerPixel() + i  ], di ),
    //                       lerp( _images[ faceIndex ][ ( j1 * size + i0 ) * getSamplePerPixel() + i  ],
    //                             _images[ faceIndex ][ ( j1 * size + i1 ) * getSamplePerPixel() + i  ], di ), dj );

    color[0] = _images[ faceIndex ][ ( j0 * size + i0 ) * getSamplePerPixel()     ];
    color[1] = _images[ faceIndex ][ ( j0 * size + i0 ) * getSamplePerPixel() + 1 ];
    color[2] = _images[ faceIndex ][ ( j0 * size + i0 ) * getSamplePerPixel() + 2 ];


    //std::cout << "face " << index << " color " << r << " " << g << " " << b << std::endl;
}


std::string getOutputImageFilename(int level, int index, const std::string& output) {
    std::stringstream ss;
    ss << output << "fixup_" << level << "_" << index << ".tif";
    return ss.str();
}
