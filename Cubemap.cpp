#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

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
    _size = 0;
    for ( int i = 0; i < 6; i++ ) {
        _images[i] = 0;
    }
}

Cubemap::~Cubemap()
{
    for ( int i = 0; i < 6; i++ ) {
        if ( _images[i] )
            delete [] _images[i];
    }
}

void Cubemap::init( int size, int sample )
{
    _size = size;
    _samplePerPixel = sample;

    for ( int i = 0; i < 6; i++ ) {
        if (_images[i])
            delete [] _images[i];
        _images[i] = new float[size*size*sample];
    }
}

void Cubemap::fill( const Vec4f& fill )
{
    for ( int i = 0; i < 6; i++ ) {
        if (_images[i]) {

            if ( _samplePerPixel > 3 ) {
                for ( int j = 0; j < _size*_size; j += _samplePerPixel ) {
                    _images[i][j] = fill[0];
                    _images[i][j+1] = fill[1];
                    _images[i][j+2] = fill[2];
                    _images[i][j+3] = fill[3];
                }

            } else {

                for ( int j = 0; j < _size*_size; j += _samplePerPixel ) {
                    _images[i][j] = fill[0];
                    _images[i][j+1] = fill[1];
                    _images[i][j+2] = fill[2];
                }

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

void Cubemap::sample(const Vec3f& direction, Vec3f& color ) const {


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

    double sc = CubemapFace[index][0] * direction;
    double tc = CubemapFace[index][1] * direction;

    // map into uv (st) [0, 1]
    double s = (sc * maInv + 1.0) * 0.5;
    double t = (tc * maInv + 1.0) * 0.5;

    int px = int( floor( s * _size ) ); // width == height
    int py = int( floor( t * _size ) ); // width == height

    //std::cout << " px " << px << " py " << py << std::endl;

    // it's not a nearest even a bilinear
    int indexPixel = ( py * _size + px ) * _samplePerPixel;
    float r = _images[ index ][ indexPixel ];
    float g = _images[ index ][ indexPixel + 1 ];
    float b = _images[ index ][ indexPixel + 2 ];
    color[0] = r;
    color[1] = g;
    color[2] = b;
    //std::cout << "face " << index << " color " << r << " " << g << " " << b << std::endl;
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

float Cubemap::texelCoordSolidAngle(int faceIdx, float aU, float aV)
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


void Cubemap::buildNormalizerSolidAngleCubemap(int size, int fixup)
{

    init(size, 4);
    int iCubeFace, u, v;

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

    int srcSize = srcCubemap->_size;
    int dstSize = dstCubemap->_size;

    //pointers used to walk across the image surface
    float *normCubeRowStartPtr;
    float *srcCubeRowStartPtr;
    float *dstCubeRowStartPtr;
    float *texelVect;

    const int srcCubeMapNumChannels	= srcCubemap->_samplePerPixel;
    const int dstCubeMapNumChannels	= dstCubemap->_samplePerPixel; //DstCubeImage[0].m_NumChannels;

    //First step - Generate SH coefficient for the diffuse convolution

    //Regenerate normalization cubemap
    //clear pre-existing normalizer cube map
    // for(int iCubeFace=0; iCubeFace<6; iCubeFace++)
    // {
    // 	m_NormCubeMap[iCubeFace].Clear();
    // }

    Cubemap normCubemap = Cubemap();

    //Normalized vectors per cubeface and per-texel solid angle
    normCubemap.buildNormalizerSolidAngleCubemap(srcCubemap->_size, fixup);

    const int normCubeMapNumChannels = normCubemap._samplePerPixel; // This need to be init here after the generation of m_NormCubeMap

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
            normCubeRowStartPtr = &normCubemap._images[iFaceIdx][ normCubeMapNumChannels * (y * srcSize)];
            srcCubeRowStartPtr	= &srcCubemap->_images[iFaceIdx][ srcCubeMapNumChannels * (y * srcSize)];

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
    normCubemap.buildNormalizerSolidAngleCubemap(dstCubemap->_size, fixup );


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
            normCubeRowStartPtr = &normCubemap._images[iFaceIdx][normCubeMapNumChannels * (y * dstSize)];
            dstCubeRowStartPtr	= &dstCubemap->_images[iFaceIdx][dstCubeMapNumChannels * (y * dstSize)];

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


void Cubemap::write( const std::string& filename )
{
    ImageOutput* out = ImageOutput::create (filename);

    // Use Create mode for the first level.
    ImageOutput::OpenMode appendmode = ImageOutput::Create;

    // Write the individual subimages
    for (int s = 0;  s < 6;  ++s) {
        ImageSpec spec( _size, _size, _samplePerPixel, TypeDesc::FLOAT);
        out->open (filename, spec, appendmode);
        out->write_image (TypeDesc::FLOAT, _images[s] );
        // Use AppendSubimage mode for subsequent levels
        appendmode = ImageOutput::AppendSubimage;
    }
    out->close ();
    delete out;
}


bool Cubemap::loadCubemap(const std::string& name)
{
    ImageInput* input = ImageInput::open ( name );

    for ( int i = 0; i < 6; i++) {
        ImageSpec spec;
        input->seek_subimage(i, 0, spec );

        if ( !_size ) {

            if ( spec.nchannels <3 ) {
                std::cout << "error your cubemap should have at least 3 channels" << std::endl;
                return false;
            }
            init( spec.width, spec.nchannels );
        }

        if ( spec.width != spec.height && spec.width != _size ) {
            std::cout << "Size of sub image " << i << " is not correct" << std::endl;
            return false;
        }
        input->read_image( TypeDesc::FLOAT, _images[i]);
    }
    input->close();
    delete input;
    return true;
}

void Cubemap::computePrefilteredEnvironment( const std::string& output, int startSize, int endSize, uint nbSamples ) {

    int computeStartSize = startSize;
    if (!computeStartSize)
        computeStartSize = _size;

    int totalMipmap = log2(computeStartSize);
    int endMipMap = totalMipmap - log2( endSize );

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
        ss << output << "_" << size << ".tif";

        // generate debug color cubemap after limit size
        if ( i <= endMipMap ) {
            std::cout << "compute level " << i << " with roughness " << roughnessLinear << " " << size << " x " << size << " to " << ss.str();
            cubemap.computePrefilterCubemapAtLevel( roughnessLinear, *this, nbSamples);
        } else {
            cubemap.fill(Vec4f(1.0,0.0,1.0,1.0));
        }
        cubemap.write( ss.str() );
    }
}

void Cubemap::computePrefilterCubemapAtLevel( float roughnessLinear, const Cubemap& inputCubemap, uint nbSamples ) {
    iterateOnFace(0, roughnessLinear, inputCubemap, nbSamples);
    iterateOnFace(1, roughnessLinear, inputCubemap, nbSamples);
    iterateOnFace(2, roughnessLinear, inputCubemap, nbSamples);
    iterateOnFace(3, roughnessLinear, inputCubemap, nbSamples);
    iterateOnFace(4, roughnessLinear, inputCubemap, nbSamples);
    iterateOnFace(5, roughnessLinear, inputCubemap, nbSamples);
}


void Cubemap::iterateOnFace( int face, float roughnessLinear, const Cubemap& cubemap, uint nbSamples ) {

    // more the roughness is and more the solid angle is big
    // so we want to adapt the number of sample depends on roughness
    // eg for a cubemap a solid angle of 180 means 3 times the pixel area
    // size*size*3 is the maximum sample
    uint numSamples = 1 << uint(floor( log2(nbSamples ) ));

    if ( roughnessLinear == 0.0 )
        numSamples = 1;

    if ( face == 0 )
        std::cout << " " << numSamples << " samples" << std::endl;

    int size = _size;

    for ( int j = 0; j < size; j++ ) {
        int lineIndex = j*_samplePerPixel*size;

#pragma omp parallel
#pragma omp for
        for ( int i = 0; i < size; i++ ) {

            int index = lineIndex + i*_samplePerPixel;
            Vec3f direction;

            texelCoordToVect( face, float(i), float(j), size, &direction[0] );

#if 0
            int faceIndex = -1;
            float u , v;
            vectToTexelCoord(direction, cubemap._size, faceIndex, u,v );
            // std::cout << "u: " << u << " " << float(i) << " " << cubemap._size << std::endl;
            // std::cout << "v: " << v << " " << float(j) << " " << cubemap._size << std::endl;

            Vec3f c0;
            cubemap.getSample(direction, c0);
            const float* ptr = &cubemap._images[face][j*_samplePerPixel*cubemap._size + i*_samplePerPixel];
            // std::cout << "r: " << ptr[0] << " " << c0[0] << std::endl;
            // std::cout << "g: " << ptr[1] << " " << c0[1] << std::endl;
            // std::cout << "b: " << ptr[2] << " " << c0[2] << std::endl;

            if ( !(ptr[0] == c0[0]) ||
                 !(ptr[1] == c0[1]) ||
                 !(ptr[2] == c0[2]) ) {
                std::cout << "face: " << face << " " << faceIndex << std::endl;
                std::cout << "u: " << u << " " << float(i) << " " << cubemap._size << std::endl;
                std::cout << "v: " << v << " " << float(j) << " " << cubemap._size << std::endl;

                std::cout << "r: " << ptr[0] << " " << c0[0] << std::endl;
                std::cout << "g: " << ptr[1] << " " << c0[1] << std::endl;
                std::cout << "b: " << ptr[2] << " " << c0[2] << std::endl;

            }
            assert ( ptr[0] == c0[0] );
            assert ( ptr[1] == c0[1] );
            assert ( ptr[2] == c0[2] );
            // assert( u == float(i) );
            // assert( v == float(j) );
            // assert( faceIndex == face );

#endif
            Vec3f resultColor = cubemap.prefilterEnvMap( roughnessLinear, direction, numSamples );

            _images[face][ index     ] = resultColor[0];
            _images[face][ index + 1 ] = resultColor[1];
            _images[face][ index + 2 ] = resultColor[2];

            //std::cout << "face " << face << " processing " << i << "x" << j << std::endl;
#if 0
            //sample( direction, resultColor );
            Vec3f diff = (color - resultColor);
            if ( fabs(diff[0]) > 1e-6 || fabs(diff[1]) > 1e-6 || fabs(diff[2]) > 1e-6 ) {
                std::cout << "face " << face << " " << i << "x" << j << " color error " << diff[0] << " " << diff[1] << " " << diff[2] << std::endl;
                std::cout << "direction " << direction[0] << " " << direction[1] << " " << direction[2]  << std::endl;
                return;
            }
#endif

        }
    }
}


Vec3f Cubemap::prefilterEnvMap( float roughnessLinear, const Vec3f& R, const uint numSamples2 ) const {
    Vec3f N = R;
    Vec3f V = R;
    Vec3d prefilteredColor = Vec3d(0,0,0);

    double totalWeight = 0;
    Vec3f color;

    Vec3f UpVector = fabs(N[2]) < 0.999 ? Vec3f(0,0,1) : Vec3f(1,0,0);
    Vec3f TangentX = normalize( cross( UpVector, N ) );
    Vec3f TangentY = normalize( cross( N, TangentX ) );

    unsigned int numSamples = numSamples2;

    //for( uint p = 0; p < 3; p++ )
        for( uint i = 0; i < numSamples; i++ ) {
            Vec2f Xi = hammersley( i, numSamples );

            // importance sampling

            // float Phi = PI2 * Xi[0];
            // float CosTheta = sqrt( (1.0 - Xi[1]) / ( 1.0 + a2min1 * Xi[1] ) );
            // float SinTheta = sqrt( 1.0 - std::min(1.0f, CosTheta * CosTheta ) );
            // Vec3f H;
            // H[0] = SinTheta * cos( Phi );
            // H[1] = SinTheta * sin( Phi );
            // H[2] = CosTheta;

            // Tangent to world space
            Vec3f H =  importanceSampleGGX( Xi, roughnessLinear, N, TangentX, TangentY);
            H.normalize();

            Vec3f L =  H * ( dot( V, H ) * 2.0 ) - V;
            L.normalize();
            float NoL = saturate( dot( N, L ) );

            if( NoL > 0.0 ) {
                getSample( L, color );
                prefilteredColor += Vec3d( color * NoL );
                totalWeight += NoL;
            }
        }

        return prefilteredColor / totalWeight;
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

void Cubemap::getSample(const Vec3f& direction, Vec3f& color ) const {

    float u,v;
    int faceIndex;

    int size = _size;
    vectToTexelCoord(direction, size, faceIndex, u,v );


    const float ii = clamp(u - 0.5f, 0.0f, size - 1.0f);
    const float jj = clamp(v - 0.5f, 0.0f, size - 1.0f);

    // const long  i0 = lrintf(ii);
    // const long  j0 = lrintf(jj);

    const long  i0 = lrintf(u);
    const long  j0 = lrintf(v);


    // for ( int i = 0; i < 3; i++ )
    //     color[i] = lerp( lerp( _images[ faceIndex ][ ( j0 * size + i0 ) * _samplePerPixel + i  ],
    //                             _images[ faceIndex ][ ( j0 * size + i1 ) * _samplePerPixel + i  ], di ),
    //                       lerp( _images[ faceIndex ][ ( j1 * size + i0 ) * _samplePerPixel + i  ],
    //                             _images[ faceIndex ][ ( j1 * size + i1 ) * _samplePerPixel + i  ], di ), dj );

    color[0] = _images[ faceIndex ][ ( j0 * size + i0 ) * _samplePerPixel     ];
    color[1] = _images[ faceIndex ][ ( j0 * size + i0 ) * _samplePerPixel + 1 ];
    color[2] = _images[ faceIndex ][ ( j0 * size + i0 ) * _samplePerPixel + 2 ];


    //std::cout << "face " << index << " color " << r << " " << g << " " << b << std::endl;
}


std::string getOutputImageFilename(int level, int index, const std::string& output) {
    std::stringstream ss;
    ss << output << "fixup_" << level << "_" << index << ".tif";
    return ss.str();
}


Cubemap* Cubemap::makeSeamless() const
{
    #define DEBUG
    const std::string& output = "/tmp/debug_cubemap_fixBorder";
    static int level = 0;

    const Cubemap& cm = *this;


    int size = cm._size;
    int fixSize = 1;

    // if the texture is 2 or less no need to copy edge
    // it will be fixed with corner for tex size 2x2
    // and we make the same pixel for all cubemap face for tex size = 1
    if ( size <= 2 ) // dont fix edge for size <= 2 need special operations
        fixSize = 0;


    int targetSize = size;
    int sizeWithoutBorder = targetSize - fixSize*2;

    std::cout << "fix border on cubemap to "  << size << " x " << size << " x " << 3 << std::endl;

    Cubemap* dst = new Cubemap();
    dst->init( size );


    ImageBuf* original[6];
    ImageBuf* resized[6];

    // resize each image less 1 border pixel on each eadge
    for ( int i = 0; i < 6; i++) {

        ImageSpec specOriginal( size, size, cm._samplePerPixel, TypeDesc::FLOAT );
        specOriginal.attribute("oiio:ColorSpace", "Linear");
        ImageBuf* imageOriginal = new ImageBuf( specOriginal, cm._images[i] );

        // Copy the first 3 channels of an RGBA, drop the alpha
        ImageBuf* RGB = imageOriginal;
        if ( cm._samplePerPixel != 3 ) {
            ImageSpec specRGB( size, size, 3, TypeDesc::FLOAT );
            specRGB.attribute("oiio:ColorSpace", "Linear");
            RGB = new ImageBuf( specRGB );
            ImageBufAlgo::channels (*RGB, *imageOriginal, 3, 0, 0 );
        }

        ImageSpec specResize( sizeWithoutBorder, sizeWithoutBorder, 3, TypeDesc::FLOAT );
        specResize.attribute("oiio:ColorSpace", "Linear");
        ImageBuf* imageResized = new ImageBuf( specResize);

        ImageBufAlgo::resize(*imageResized, *RGB );

        resized[i] = imageResized;
        original[i] = RGB;
    }

    int subSize = sizeWithoutBorder;


    // for the pictures
    // http://scalibq.wordpress.com/2013/06/23/cubemaps/

    // compute 8 corner

    // corner X+ top left
    Vec3f tmp;
    Vec3f corner0(0.0,0.0,0.0);
    {
        // z+ top right
        original[4]->getpixel(size-1, 0, 0, &tmp[0] );
        corner0 += tmp;

        // x+ top left
        original[0]->getpixel(0, 0, 0, &tmp[0] );
        corner0 += tmp;

        // y+ bot right
        original[2]->getpixel(size-1, size-1, 0, &tmp[0] );
        corner0 += tmp;

    }
    corner0 /= 3.0;


    // corner X+ top right
    Vec3f corner1(0,0,0);
    {
        // z- top left
        original[5]->getpixel(0, 0, 0, &tmp[0] );
        corner1 += tmp;

        // x+ top right
        original[0]->getpixel(size -1 , 0, 0, &tmp[0] );
        corner1 += tmp;

        // y+ top right
        original[2]->getpixel(size-1, 0, 0, &tmp[0] );
        corner1 += tmp;
    }
    corner1 /= 3.0;


    // corner X+ bot right
    Vec3f corner2(0,0,0);
    {
        // z- bot left
        original[5]->getpixel(0, size-1, 0, &tmp[0] );
        corner2 += tmp;

        // x+ bot right
        original[0]->getpixel(size-1 , size-1, 0, &tmp[0] );
        corner2 += tmp;

        // y- bot right
        original[3]->getpixel(size-1, size-1, 0, &tmp[0] );
        corner2 += tmp;
    }
    corner2 /= 3.0;


    // corner X+ bot left
    Vec3f corner3(0,0,0);
    {
        // z+ bot right
        original[4]->getpixel(size-1, size-1, 0, &tmp[0] );
        corner3 += tmp;

        // x+ bot left
        original[0]->getpixel(0 , size-1, 0, &tmp[0] );
        corner3 += tmp;

        // y- top right
        original[3]->getpixel(size-1, 0, 0, &tmp[0] );
        corner3 += tmp;
    }
    corner3 /= 3.0;


    // corner X- top left
    Vec3f corner4(0,0,0);
    {
        // z- top right
        original[5]->getpixel(size-1, 0, 0, &tmp[0] );
        corner4 += tmp;

        // x- top left
        original[1]->getpixel(0, 0, 0, &tmp[0] );
        corner4 += tmp;

        // y+ top left
        original[2]->getpixel( 0, 0, 0, &tmp[0] );
        corner4 += tmp;
    }
    corner4 /= 3.0;


    // corner X- top right
    Vec3f corner5(0,0,0);
    {
        // z+ top left
        original[4]->getpixel(0, 0, 0, &tmp[0] );
        corner5 += tmp;

        // x- top right
        original[1]->getpixel(size-1, 0, 0, &tmp[0] );
        corner5 += tmp;

        // y+ bot right
        original[2]->getpixel( 0, size-1, 0, &tmp[0] );
        corner5 += tmp;
    }
    corner5 /= 3.0;


    // corner X- bot right
    Vec3f corner6(0,0,0);
    {
        // z+ bot left
        original[4]->getpixel(0, size-1, 0, &tmp[0] );
        corner6 += tmp;

        // x- bot right
        original[1]->getpixel(size-1, size-1, 0, &tmp[0] );
        corner6 += tmp;

        // y- top left
        original[3]->getpixel( 0, 0, 0, &tmp[0] );
        corner6 += tmp;
    }
    corner6 /= 3.0;


    // corner X- bot left
    Vec3f corner7(0,0,0);
    {
        // z- bot left
        original[5]->getpixel(size-1, size-1, 0, &tmp[0] );
        corner7 += tmp;

        // x- bot left
        original[1]->getpixel(0, size-1, 0, &tmp[0] );
        corner7 += tmp;

        // y- top left
        original[3]->getpixel( 0, size-1, 0, &tmp[0] );
        corner7 += tmp;
    }
    corner7 /= 3.0;


    // special case for mipmap 1
    // texture size = 1
    float allFace[3]; allFace[0] = 0; allFace[1] = 0; allFace[2] = 0;
    if ( size == 1 ) {
        for ( int i = 0; i < 6; i++ ) {
            float tmp[3];
            resized[i]->getpixel( 0, 0, 0, &tmp[0] );
            allFace[0] += tmp[0]; allFace[1] += tmp[1]; allFace[2] += tmp[2];
        }
        allFace[0] /= 6.0; allFace[1] /= 6.0; allFace[2] /= 6.0;
    }

    // x
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 0, output).c_str(), specOut, dst->_images[0] );
        ImageBufAlgo::paste (imageBufOut, fixSize, fixSize, 0, 0, *resized[0] );

        //  top edge: y positif
        ImageBuf spanTop;
        ImageBufAlgo::rotate90(spanTop, *resized[2]);
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, fixSize, i , 0, 0,
                                 spanTop, ROI(0, subSize, subSize-1, subSize));
        }

        //  right edge: z negatif
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, subSize+fixSize+i, fixSize , 0, 0,
                                 *resized[5], ROI(0 ,1, 0, subSize));
        }


        //  left edge: z negatif
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, i, fixSize , 0, 0,
                                 *resized[4], ROI( subSize-1 ,subSize, 0, subSize));
        }

        //  bottom edge: y negatif
        ImageBuf spanBot;
        ImageBufAlgo::rotate270(spanBot, *resized[3]);
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, fixSize, subSize + fixSize + i , 0, 0,
                                 spanBot, ROI(0, subSize, 0, 1));
        }

        // corners
        imageBufOut.setpixel( 0,0,0, &corner0[0] );
        imageBufOut.setpixel( size-1,0,0, &corner1[0] );
        imageBufOut.setpixel( size-1,size-1,0, &corner2[0] );
        imageBufOut.setpixel( 0,size-1,0, &corner3[0] );

        // when last mipmap level
        if ( size == 1 )
            imageBufOut.setpixel(0,0,0, allFace);

#ifdef DEBUG
        imageBufOut.save();
#endif
    }


    // -x
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 1, output).c_str(), specOut, dst->_images[1]);
        ImageBufAlgo::paste (imageBufOut, fixSize, fixSize, 0, 0, *resized[1] );

        //  top edge: y positif
        ImageBuf spanTop;
        ImageBufAlgo::rotate270(spanTop, *resized[2]);
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, fixSize, i , 0, 0,
                                 spanTop, ROI(0, subSize, subSize-1, subSize));
        }

        //  right edge: z negatif
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, subSize+fixSize+i, fixSize , 0, 0,
                                 *resized[4], ROI(0 ,1, 0, subSize));
        }


        //  left edge: z negatif
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, i, fixSize , 0, 0,
                                 *resized[5], ROI( subSize-1 ,subSize, 0, subSize));
        }

        //  bottom edge: y negatif
        ImageBuf spanBot;
        ImageBufAlgo::rotate90(spanBot, *resized[3]);
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, fixSize, subSize + fixSize + i , 0, 0,
                                 spanBot, ROI(0, subSize, 0, 1));
        }

        // &corners[0]
        imageBufOut.setpixel( 0,0,0, &corner4[0] );
        imageBufOut.setpixel( size-1,0,0, &corner5[0] );
        imageBufOut.setpixel( size-1,size-1,0, &corner6[0] );
        imageBufOut.setpixel( 0,size-1,0, &corner7[0] );

        // when last mipmap level
        if ( size == 1 )
            imageBufOut.setpixel(0,0,0, allFace);

#ifdef DEBUG
        imageBufOut.save();
#endif
    }


    // y
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 2, output).c_str(), specOut, dst->_images[2]);
        ImageBufAlgo::paste (imageBufOut, fixSize, fixSize, 0, 0, *resized[2] );

        //  top edge: y positif
        {
            ImageBuf span;
            ImageBufAlgo::flop(span, *resized[5]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, fixSize, i , 0, 0,
                                     span, ROI(0, subSize, 0, 1));
            }
        }

        //  right edge: x pos
        {
            ImageBuf span;
            ImageBufAlgo::rotate270(span, *resized[0]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, subSize+fixSize+i, fixSize , 0, 0,
                                     span, ROI(0 ,1, 0, subSize));
            }
        }


        //  left edge: x neg
        {
            ImageBuf span;
            ImageBufAlgo::rotate90(span, *resized[1]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, i, fixSize , 0, 0,
                                     span, ROI(subSize-1 ,subSize, 0, subSize));
            }
        }

        //  bottom edge: z positif
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, fixSize, subSize + fixSize + i , 0, 0,
                                 *resized[4], ROI(0, subSize, 0, 1));
        }

        // &corners[0]
        imageBufOut.setpixel( 0,0,0, &corner4[0] );
        imageBufOut.setpixel( size-1,0,0, &corner1[0] );
        imageBufOut.setpixel( size-1,size-1,0, &corner0[0] );
        imageBufOut.setpixel( 0,size-1,0, &corner5[0] );

        // when last mipmap level
        if ( size == 1 )
            imageBufOut.setpixel(0,0,0, allFace);

#ifdef DEBUG
        imageBufOut.save();
#endif
    }


    // -y
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 3, output).c_str(), specOut, dst->_images[3]);
        ImageBufAlgo::paste (imageBufOut, fixSize, fixSize, 0, 0, *resized[3] );

        //  top edge: z positif
        {
            ImageBuf span;
            //ImageBufAlgo::flop(span, *resized[4]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, fixSize, i , 0, 0,
                                     *resized[4], ROI(0, subSize, subSize-1, subSize));
            }
        }

        //  right edge: x pos
        {
            ImageBuf span;
            ImageBufAlgo::rotate90(span, *resized[0]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, subSize+fixSize+i, fixSize , 0, 0,
                                     span, ROI(0 ,1, 0, subSize));
            }
        }


        //  left edge: x neg
        {
            ImageBuf span;
            ImageBufAlgo::rotate270(span, *resized[1]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, i, fixSize , 0, 0,
                                     span, ROI(subSize-1 ,subSize, 0, subSize));
            }
        }

        //  bottom edge: z neg
        {
            ImageBuf span;
            ImageBufAlgo::flop(span, *resized[5]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, fixSize, subSize + fixSize + i , 0, 0,
                                     span, ROI(0, subSize, subSize-1, subSize));
            }
        }

        imageBufOut.setpixel( 0,0,0, &corner6[0] );
        imageBufOut.setpixel( size-1,0,0, &corner3[0] );
        imageBufOut.setpixel( size-1,size-1,0, &corner2[0] );
        imageBufOut.setpixel( 0,size-1,0, &corner7[0] );

        // when last mipmap level
        if ( size == 1 )
            imageBufOut.setpixel(0,0,0, allFace);

#ifdef DEBUG
        imageBufOut.save();
#endif
    }


    // z
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 4, output).c_str(), specOut, dst->_images[4]);
        ImageBufAlgo::paste (imageBufOut, fixSize, fixSize, 0, 0, *resized[4] );

        //  top edge: y pos
        {
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, fixSize, i , 0, 0,
                                     *resized[2], ROI(0, subSize, subSize-1, subSize));
            }
        }

        //  right edge: x pos
        {
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, subSize+fixSize+i, fixSize , 0, 0,
                                     *resized[0], ROI(0 ,1, 0, subSize));
            }
        }


        //  left edge: x neg
        {
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, i, fixSize , 0, 0,
                                     *resized[1], ROI(subSize-1 ,subSize, 0, subSize));
            }
        }

        //  bottom edge: y neg
        for ( int i = 0; i < fixSize; i++ ) {
            ImageBufAlgo::paste (imageBufOut, fixSize, subSize + fixSize + i , 0, 0,
                                 *resized[3], ROI(0, subSize, 0, 1));
        }


        imageBufOut.setpixel( 0,0,0, &corner5[0] );
        imageBufOut.setpixel( size-1,0,0, &corner0[0] );
        imageBufOut.setpixel( size-1,size-1,0, &corner3[0] );
        imageBufOut.setpixel( 0,size-1,0, &corner6[0] );

        // when last mipmap level
        if ( size == 1 )
            imageBufOut.setpixel(0,0,0, allFace);

#ifdef DEBUG
        imageBufOut.save();
#endif
    }



    // -z
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 5, output).c_str(), specOut, dst->_images[5]);
        ImageBufAlgo::paste (imageBufOut, fixSize, fixSize, 0, 0, *resized[5] );

        //  top edge: y pos
        {
            ImageBuf span;
            ImageBufAlgo::flop(span, *resized[2]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, fixSize, i , 0, 0,
                                     span, ROI(0, subSize, 0,1));
            }
        }

        //  right edge: x neg
        {
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, subSize+fixSize+i, fixSize , 0, 0,
                                     *resized[1], ROI(0 ,1, 0, subSize));
            }
        }


        //  left edge: x pos
        {
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, i, fixSize , 0, 0,
                                     *resized[0], ROI(subSize-1 ,subSize, 0, subSize));
            }
        }

        //  bottom edge: y neg
        {
            ImageBuf span;
            ImageBufAlgo::flop(span, *resized[3]);
            for ( int i = 0; i < fixSize; i++ ) {
                ImageBufAlgo::paste (imageBufOut, fixSize, subSize + fixSize + i , 0, 0,
                                     span, ROI(0, subSize, subSize-1, subSize));
            }
        }

        imageBufOut.setpixel( 0,0,0, &corner1[0] );
        imageBufOut.setpixel( size-1,0,0, &corner4[0] );
        imageBufOut.setpixel( size-1,size-1,0, &corner7[0] );
        imageBufOut.setpixel( 0,size-1,0, &corner2[0] );

        // when last mipmap level
        if ( size == 1 )
            imageBufOut.setpixel(0,0,0, allFace);

#ifdef DEBUG
        imageBufOut.save();
#endif
    }

    for ( int i = 0; i < 6; i++) {
        delete resized[i];
    }

    level++;

    return dst;
}
