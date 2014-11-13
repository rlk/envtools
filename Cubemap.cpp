#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <tiffio.h>

#include "Cubemap"
#include "sRGB.h"
#include "gray.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filter.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
OIIO_NAMESPACE_USING


#define CP_UDIR     0
#define CP_VDIR     1
#define CP_FACEAXIS 2



Cubemap::Cubemap()
{
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

void Cubemap::init( int size, int sample, int bits)
{
    _size = size;
    _samplePerPixel = sample;
    _bitsPerSample = bits;

    for ( int i = 0; i < 6; i++ ) {
        if (_images[i])
            delete [] _images[i];
        _images[i] = new float[size*size*sample];
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

//int a_FaceIdx, float a_U, float a_V, int a_Size, Vec3f& a_XYZ, FixUpType a_FixupType
void Cubemap::texelCoordToVect(int faceIdx, float U, float V, float* resultXYZ, FixUpType fixupType)
{
    float nvcU, nvcV;
    Vec3f XYZ;

    if (fixupType == CP_FIXUP_STRETCH && _size > 1)
    {
        // Code from Nvtt : http://code.google.com/p/nvidia-texture-tools/source/browse/trunk/src/nvtt/CubeSurface.cpp
        // transform from [0..res - 1] to [-1 .. 1], match up edges exactly.
        nvcU = (2.0f * (float)U / ((float)_size - 1.0f) ) - 1.0f;
        nvcV = (2.0f * (float)V / ((float)_size - 1.0f) ) - 1.0f;
    }
    else
    {
        // Change from original AMD code
        // transform from [0..res - 1] to [- (1 - 1 / res) .. (1 - 1 / res)]
        // + 0.5f is for texel center addressing
        nvcU = (2.0f * ((float)U + 0.5f) / (float)_size ) - 1.0f;
        nvcV = (2.0f * ((float)V + 0.5f) / (float)_size ) - 1.0f;
    }

    if (fixupType == CP_FIXUP_WARP && _size > 1)
    {
        // Code from Nvtt : http://code.google.com/p/nvidia-texture-tools/source/browse/trunk/src/nvtt/CubeSurface.cpp
        float a = powf(float(_size), 2.0f) / powf(float(_size - 1), 3.0f);
        nvcU = a * powf(nvcU, 3) + nvcU;
        nvcV = a * powf(nvcV, 3) + nvcV;

        // Get current vector
        //generate x,y,z vector (xform 2d NVC coord to 3D vector)
        //U contribution
        //VM_SCALE3(XYZ, CubemapFace[faceIdx][CP_UDIR], nvcU);
        XYZ = CubemapFace[faceIdx][CP_UDIR] * nvcU;

        //V contribution
        //VM_SCALE3(tempVec, CubemapFace[faceIdx][CP_VDIR], nvcV);
        //VM_ADD3(XYZ, tempVec, XYZ);
        XYZ += CubemapFace[faceIdx][CP_VDIR] * nvcV;

        //add face axis
        //VM_ADD3(XYZ, CubemapFace[faceIdx][CP_FACEAXIS], XYZ);
        XYZ += CubemapFace[faceIdx][CP_FACEAXIS];


        //normalize vector
        //VM_NORM3(XYZ, XYZ);
        XYZ.normalize();

    }
    else if (fixupType == CP_FIXUP_BENT && _size > 1)
    {
        // Method following description of Physically based rendering slides from CEDEC2011 of TriAce

        // Get vector at edge
        Vec3f EdgeNormalU;
        Vec3f EdgeNormalV;
        Vec3f EdgeNormal;
        Vec3f EdgeNormalMinusOne;

        // Recover vector at edge
        //U contribution
        //VM_SCALE3(EdgeNormalU, CubemapFace[faceIdx][CP_UDIR], nvcU < 0.0 ? -1.0f : 1.0f);
        EdgeNormalU = CubemapFace[faceIdx][CP_UDIR] * ( nvcU < 0.0 ? -1.0f : 1.0f);

        //V contribution
        //VM_SCALE3(EdgeNormalV, CubemapFace[faceIdx][CP_VDIR], nvcV < 0.0 ? -1.0f : 1.0f);
        EdgeNormalV = CubemapFace[faceIdx][CP_VDIR] * (nvcV < 0.0 ? -1.0f : 1.0f);

        //VM_ADD3(EdgeNormal, EdgeNormalV, EdgeNormalU);
        EdgeNormal += EdgeNormalV;
        EdgeNormal += EdgeNormalU;

        //add face axis
        //VM_ADD3(EdgeNormal, CubemapFace[faceIdx][CP_FACEAXIS], EdgeNormal);
        EdgeNormal += CubemapFace[faceIdx][CP_FACEAXIS];

        //normalize vector
        //VM_NORM3(EdgeNormal, EdgeNormal);
        EdgeNormal.normalize();

        // Get vector at (edge - 1)
        float nvcUEdgeMinus1 = (2.0f * ((float)(nvcU < 0.0f ? 0 : _size-1) + 0.5f) / (float)_size ) - 1.0f;
        float nvcVEdgeMinus1 = (2.0f * ((float)(nvcV < 0.0f ? 0 : _size-1) + 0.5f) / (float)_size ) - 1.0f;

        // Recover vector at (edge - 1)
        //U contribution
        //VM_SCALE3(EdgeNormalU, CubemapFace[faceIdx][CP_UDIR], nvcUEdgeMinus1);
        EdgeNormalU = CubemapFace[faceIdx][CP_UDIR] * nvcUEdgeMinus1;

        //V contribution
        //VM_SCALE3(EdgeNormalV, CubemapFace[faceIdx][CP_VDIR], nvcVEdgeMinus1);
        EdgeNormalV = CubemapFace[faceIdx][CP_VDIR] * nvcVEdgeMinus1;

        //VM_ADD3(EdgeNormalMinusOne, EdgeNormalV, EdgeNormalU);
        EdgeNormalMinusOne += EdgeNormalV;
        EdgeNormalMinusOne += EdgeNormalU;

        //add face axis
        //VM_ADD3(EdgeNormalMinusOne, CubemapFace[faceIdx][CP_FACEAXIS], EdgeNormalMinusOne);
        EdgeNormalMinusOne += CubemapFace[faceIdx][CP_FACEAXIS];

        //normalize vector
        //VM_NORM3(EdgeNormalMinusOne, EdgeNormalMinusOne);
        EdgeNormalMinusOne.normalize();

        // Get angle between the two vector (which is 50% of the two vector presented in the TriAce slide)
        //float AngleNormalEdge = acosf(VM_DOTPROD3(EdgeNormal, EdgeNormalMinusOne));
        float AngleNormalEdge = acosf( dot(EdgeNormal, EdgeNormalMinusOne) );

        // Here we assume that high resolution required less offset than small resolution (TriAce based this on blur radius and custom value)
        // Start to increase from 50% to 100% target angle from 128x128x6 to 1x1x6
        float NumLevel = (logf(std::min(_size, 128))  / logf(2)) - 1;
        AngleNormalEdge = lerp<float>(0.5f * AngleNormalEdge, AngleNormalEdge, 1.0f - (NumLevel/6.f) );

        float factorU = fabs((2.0f * ((float)U) / (float)(_size - 1) ) - 1.0f);
        float factorV = fabs((2.0f * ((float)V) / (float)(_size - 1) ) - 1.0f);
        AngleNormalEdge = lerp<float>(0.0f, AngleNormalEdge, std::max(factorU, factorV) );

        // Get current vector
        //generate x,y,z vector (xform 2d NVC coord to 3D vector)
        //U contribution
        //VM_SCALE3(XYZ, CubemapFace[faceIdx][CP_UDIR], nvcU);
        XYZ = CubemapFace[faceIdx][CP_UDIR] * nvcU;

        //V contribution
        //VM_SCALE3(tempVec, CubemapFace[faceIdx][CP_VDIR], nvcV);
        //VM_ADD3(XYZ, tempVec, XYZ);
        XYZ += CubemapFace[faceIdx][CP_VDIR] * nvcV;

        //add face axis
        //VM_ADD3(XYZ, CubemapFace[faceIdx][CP_FACEAXIS], XYZ);
        XYZ += CubemapFace[faceIdx][CP_FACEAXIS];

        //normalize vector
        //VM_NORM3(XYZ, XYZ);
        XYZ.normalize();

        float RadiantAngle = AngleNormalEdge;
        // Get angle between face normal and current normal. Used to push the normal away from face normal.
        //float AngleFaceVector = acosf(VM_DOTPROD3(CubemapFace[faceIdx][CP_FACEAXIS], XYZ));
        float AngleFaceVector = acosf( dot(CubemapFace[faceIdx][CP_FACEAXIS], XYZ ));

        // Push the normal away from face normal by an angle of RadiantAngle
        slerp(XYZ, CubemapFace[faceIdx][CP_FACEAXIS], XYZ, 1.0f + RadiantAngle / AngleFaceVector);
    }
    else
    {
        //generate x,y,z vector (xform 2d NVC coord to 3D vector)
        //U contribution
        //VM_SCALE3(XYZ, CubemapFace[faceIdx][CP_UDIR], nvcU);
        XYZ = CubemapFace[faceIdx][CP_UDIR] * nvcU;

        //V contribution
        //VM_SCALE3(tempVec, CubemapFace[faceIdx][CP_VDIR], nvcV);
        //VM_ADD3(XYZ, tempVec, XYZ);
        XYZ += CubemapFace[faceIdx][CP_VDIR] * nvcV;

        //add face axis
        //VM_ADD3(XYZ, CubemapFace[faceIdx][CP_FACEAXIS], XYZ);
        XYZ += CubemapFace[faceIdx][CP_FACEAXIS];

        //normalize vector
        //VM_NORM3(XYZ, XYZ);
        XYZ.normalize();
    }

    resultXYZ[0] = XYZ[0];
    resultXYZ[1] = XYZ[1];
    resultXYZ[2] = XYZ[2];
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


void Cubemap::buildNormalizerSolidAngleCubemap(int size, FixUpType fixupType)
{

    init(size);
    int iCubeFace, u, v;

    //iterate over cube faces
    for(iCubeFace=0; iCubeFace<6; iCubeFace++)
    {

        // this step is done by init upper int this function
        //a_Surface[iCubeFace].Clear();
        //a_Surface[iCubeFace].Init(a_Size, a_Size, 4);  //First three channels for norm cube, and last channel for solid angle

        //fast texture walk, build normalizer cube map
        float *texelPtr = _images[iCubeFace];

        for(v=0; v < _size; v++)
        {
            for(u=0; u < _size; u++)
            {

                texelCoordToVect(iCubeFace, (float)u, (float)v, texelPtr, fixupType);

                //VM_SCALE3(texelPtr, texelPtr, 0.5f);
                //VM_BIAS3(texelPtr, texelPtr, 0.5f);

                *(texelPtr + 3) = texelCoordSolidAngle(iCubeFace, (float)u, (float)v);

                texelPtr += _samplePerPixel;
            }
        }
    }
}


Cubemap* Cubemap::shFilterCubeMap(bool useSolidAngleWeighting, FixUpType fixupType, int outputCubemapSize)
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
    normCubemap.buildNormalizerSolidAngleCubemap(srcCubemap->_size, fixupType);

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
    normCubemap.buildNormalizerSolidAngleCubemap(dstCubemap->_size, fixupType );


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

    TIFF* file = TIFFOpen(filename.c_str(), "w");

    if (!file)
        return;

    for ( int face = 0; face < 6; face++) {
        TIFFSetField(file, TIFFTAG_IMAGEWIDTH,      _size);
        TIFFSetField(file, TIFFTAG_IMAGELENGTH,     _size);
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

        for (int i = 0; i < _size; ++i) {
            uint8* start = (uint8*)_images[face];
            TIFFWriteScanline(file, (uint8 *) start + size * i, i, 0);
        }

        TIFFWriteDirectory(file);
    }

    TIFFClose(file);
}


bool Cubemap::loadCubemap(const std::string& name)
{
    TIFF *tif;
    tif = TIFFOpen(name.c_str(), "r");
    if (!tif)
        return false;

    uint32 width;
    uint32 height;
    uint16 bitsPerSample;
    uint16 samplePerPixel;

    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,  &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,     &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH,    &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplePerPixel);

    _size = (int) width;
    _bitsPerSample = (int) bitsPerSample;
    _samplePerPixel = (int) samplePerPixel;

    std::cout << "reading cubemap environment  6 x " << width << " x " << height << " x " << _samplePerPixel << " - " << bitsPerSample << " bits" << std::endl;
    for ( int face = 0; face < 6; ++face) {
        loadEnvFace(tif, face);
    }

    TIFFClose(tif);

    return true;
}

void Cubemap::loadEnvFace(TIFF* tif, int face)
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

    if (! (w == _size && h == _size && b == _bitsPerSample && c == _samplePerPixel)) {
        std::cerr << "can't read face " << face << std::endl;
        return;
    }

    _images[face] = new float[_size*_size*_samplePerPixel];

    /* Allocate a scanline buffer. */
    std::vector<float> s;
    s.resize(TIFFScanlineSize(tif));

    /* Iterate over all pixels of the current cube face. */
    for (uint32 i = 0; i < h; ++i) {

        if (TIFFReadScanline(tif, &s.front(), i, 0) == 1) {

            for (uint32 j = 0; j < w; ++j) {

                for (int k =0; k < _samplePerPixel; k++) {
                    float p = s[ j * c + k ];
                    _images[face][(i*w + j)*_samplePerPixel + k ] = p;
                }
            }
        }
    }
}






std::string getOutputImageFilename(int level, int index, const std::string& output) {
    std::stringstream ss;
    ss << output << "fixup_" << level << "_" << index << ".tif";
    return ss.str();
}

void fixImage(Cubemap& cm, int level, const std::string& output) {


    int size = cm._size;
    int fixSize = 1;

    if ( size <= 2 ) // dont fix edge for size <= 2 need special operations
        fixSize = 0;

    int targetSize = size;
    int sizeWithoutBorder = targetSize - fixSize*2;

    std::cout << "fix border on cubemap to "  << size << " x " << size << " x " << 3 << std::endl;

    ImageBuf* original[6];
    ImageBuf* resized[6];

    // resize each image less 1 border pixel on each eadge
    for ( int i = 0; i < 6; i++) {

        ImageSpec specOriginal( size, size, cm._samplePerPixel, TypeDesc::FLOAT );
        specOriginal.attribute("oiio:ColorSpace", "Linear");
        ImageBuf imageOriginal( specOriginal, cm._images[i] );

        // Copy the first 3 channels of an RGBA, drop the alpha
        ImageSpec specRGB( size, size, 3, TypeDesc::FLOAT );
        specRGB.attribute("oiio:ColorSpace", "Linear");
        ImageBuf* RGB = new ImageBuf( specRGB );
        ImageBufAlgo::channels (*RGB, imageOriginal, 3, NULL /*default ordering*/);

        ImageSpec specResize( sizeWithoutBorder, sizeWithoutBorder, 3, TypeDesc::FLOAT );
        specResize.attribute("oiio:ColorSpace", "Linear");
        ImageBuf* imageResized = new ImageBuf( specResize);

        ImageBufAlgo::resize(*imageResized, *RGB );

        resized[i] = imageResized;
        original[i] = RGB;
    }

    int subSize = sizeWithoutBorder;


    // compute 8 corner

    // corner X+ top left
    float tmp[3];
    float corner0[3]; corner0[0] = 0; corner0[1] = 0; corner0[2] = 0;
    {
        // z+ top right
        original[4]->getpixel(size-1, 0, 0, tmp );
        corner0[0] += tmp[0]; corner0[1] += tmp[1]; corner0[2] += tmp[2];

        // x+ top left
        original[0]->getpixel(0, 0, 0, tmp );
        corner0[0] += tmp[0]; corner0[1] += tmp[1]; corner0[2] += tmp[2];

        // y+ bot right
        original[2]->getpixel(size-1, size-1, 0, tmp );
        corner0[0] += tmp[0]; corner0[1] += tmp[1]; corner0[2] += tmp[2];
    }
    corner0[0] /= 3.0; corner0[1] /= 3.0; corner0[2] /= 3.0;


    // corner X+ top right
    float corner1[3]; corner1[0] = 0; corner1[1] = 0; corner1[2] = 0;
    {
        // z- top left
        original[5]->getpixel(0, 0, 0, tmp );
        corner1[0] += tmp[0]; corner1[1] += tmp[1]; corner1[2] += tmp[2];

        // x+ top right
        original[0]->getpixel(size -1 , 0, 0, tmp );
        corner1[0] += tmp[0]; corner1[1] += tmp[1]; corner1[2] += tmp[2];

        // y+ top right
        original[2]->getpixel(size-1, 0, 0, tmp );
        corner1[0] += tmp[0]; corner1[1] += tmp[1]; corner1[2] += tmp[2];
    }
    corner1[0] /= 3.0; corner1[1] /= 3.0; corner1[2] /= 3.0;


    // corner X+ bot right
    float corner2[3]; corner2[0] = 0; corner2[1] = 0; corner2[2] = 0;
    {
        // z- bot left
        original[5]->getpixel(0, size-1, 0, tmp );
        corner2[0] += tmp[0]; corner2[1] += tmp[1]; corner2[2] += tmp[2];

        // x+ bot right
        original[0]->getpixel(size-1 , size-1, 0, tmp );
        corner2[0] += tmp[0]; corner2[1] += tmp[1]; corner2[2] += tmp[2];

        // y- bot right
        original[3]->getpixel(size-1, size-1, 0, tmp );
        corner2[0] += tmp[0]; corner2[1] += tmp[1]; corner2[2] += tmp[2];
    }
    corner2[0] /= 3.0; corner2[1] /= 3.0; corner2[2] /= 3.0;


    // corner X+ bot left
    float corner3[3]; corner3[0] = 0; corner3[1] = 0; corner3[2] = 0;
    {
        // z+ bot right
        original[5]->getpixel(size-1, size-1, 0, tmp );
        corner3[0] += tmp[0]; corner3[1] += tmp[1]; corner3[2] += tmp[2];

        // x+ bot left
        original[0]->getpixel(0 , size-1, 0, tmp );
        corner3[0] += tmp[0]; corner3[1] += tmp[1]; corner3[2] += tmp[2];

        // y- top right
        original[3]->getpixel(size-1, 0, 0, tmp );
        corner3[0] += tmp[0]; corner3[1] += tmp[1]; corner3[2] += tmp[2];
    }
    corner3[0] /= 3.0; corner3[1] /= 3.0; corner3[2] /= 3.0;


    // corner X- top left
    float corner4[3]; corner4[0] = 0; corner4[1] = 0; corner4[2] = 0;
    {
        // z- top right
        original[5]->getpixel(size-1, 0, 0, tmp );
        corner4[0] += tmp[0]; corner4[1] += tmp[1]; corner4[2] += tmp[2];

        // x- top left
        original[1]->getpixel(0, 0, 0, tmp );
        corner4[0] += tmp[0]; corner4[1] += tmp[1]; corner4[2] += tmp[2];

        // y+ top left
        original[2]->getpixel( 0, 0, 0, tmp );
        corner4[0] += tmp[0]; corner4[1] += tmp[1]; corner4[2] += tmp[2];
    }
    corner4[0] /= 3.0; corner4[1] /= 3.0; corner4[2] /= 3.0;


    // corner X- top right
    float corner5[3]; corner5[0] = 0; corner5[1] = 0; corner5[2] = 0;
    {
        // z+ top left
        original[4]->getpixel(0, 0, 0, tmp );
        corner5[0] += tmp[0]; corner5[1] += tmp[1]; corner5[2] += tmp[2];

        // x- top right
        original[1]->getpixel(size-1, 0, 0, tmp );
        corner5[0] += tmp[0]; corner5[1] += tmp[1]; corner5[2] += tmp[2];

        // y+ bot right
        original[2]->getpixel( 0, size-1, 0, tmp );
        corner5[0] += tmp[0]; corner5[1] += tmp[1]; corner5[2] += tmp[2];
    }
    corner5[0] /= 3.0; corner5[1] /= 3.0; corner5[2] /= 3.0;


    // corner X- bot right
    float corner6[3]; corner6[0] = 0; corner6[1] = 0; corner6[2] = 0;
    {
        // z+ bot left
        original[4]->getpixel(0, size-1, 0, tmp );
        corner6[0] += tmp[0]; corner6[1] += tmp[1]; corner6[2] += tmp[2];

        // x- bot right
        original[1]->getpixel(size-1, size-1, 0, tmp );
        corner6[0] += tmp[0]; corner6[1] += tmp[1]; corner6[2] += tmp[2];

        // y- top left
        original[3]->getpixel( 0, 0, 0, tmp );
        corner6[0] += tmp[0]; corner6[1] += tmp[1]; corner6[2] += tmp[2];
    }
    corner6[0] /= 3.0; corner6[1] /= 3.0; corner6[2] /= 3.0;


    // corner X- bot left
    float corner7[3]; corner7[0] = 0; corner7[1] = 0; corner7[2] = 0;
    {
        // z- bot left
        original[5]->getpixel(size-1, size-1, 0, tmp );
        corner7[0] += tmp[0]; corner7[1] += tmp[1]; corner7[2] += tmp[2];

        // x- bot left
        original[1]->getpixel(0, size-1, 0, tmp );
        corner7[0] += tmp[0]; corner7[1] += tmp[1]; corner7[2] += tmp[2];

        // y- top left
        original[3]->getpixel( 0, size-1, 0, tmp );
        corner7[0] += tmp[0]; corner7[1] += tmp[1]; corner7[2] += tmp[2];
    }
    corner7[0] /= 3.0; corner7[1] /= 3.0; corner7[2] /= 3.0;


    // special case for mipmap 1


    // x
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 0, output).c_str(), specOut);
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
        imageBufOut.setpixel( 0,0,0, corner0 );
        imageBufOut.setpixel( size-1,0,0, corner1 );
        imageBufOut.setpixel( size-1,size-1,0, corner2 );
        imageBufOut.setpixel( 0,size-1,0, corner3 );

        imageBufOut.save();
    }


    // -x
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 1, output).c_str(), specOut);
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

        // corners
        imageBufOut.setpixel( 0,0,0, corner4 );
        imageBufOut.setpixel( size-1,0,0, corner5 );
        imageBufOut.setpixel( size-1,size-1,0, corner6 );
        imageBufOut.setpixel( 0,size-1,0, corner7 );

        imageBufOut.save();
    }


    // y
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 2, output).c_str(), specOut);
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

        // corners
        imageBufOut.setpixel( 0,0,0, corner4 );
        imageBufOut.setpixel( size-1,0,0, corner1 );
        imageBufOut.setpixel( size-1,size-1,0, corner0 );
        imageBufOut.setpixel( 0,size-1,0, corner5 );

        imageBufOut.save();
    }


    // -y
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 3, output).c_str(), specOut);
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

        imageBufOut.setpixel( 0,0,0, corner6 );
        imageBufOut.setpixel( size-1,0,0, corner3 );
        imageBufOut.setpixel( size-1,size-1,0, corner2 );
        imageBufOut.setpixel( 0,size-1,0, corner7 );

        imageBufOut.save();
    }


    // z
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 4, output).c_str(), specOut);
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


        imageBufOut.setpixel( 0,0,0, corner5 );
        imageBufOut.setpixel( size-1,0,0, corner0 );
        imageBufOut.setpixel( size-1,size-1,0, corner3 );
        imageBufOut.setpixel( 0,size-1,0, corner6 );

        imageBufOut.save();
    }



    // -z
    {
        ImageSpec specOut( targetSize, targetSize, 3, TypeDesc::FLOAT );
        ImageBuf imageBufOut( getOutputImageFilename(level, 5, output).c_str(), specOut);
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

        imageBufOut.setpixel( 0,0,0, corner1 );
        imageBufOut.setpixel( size-1,0,0, corner4 );
        imageBufOut.setpixel( size-1,size-1,0, corner7 );
        imageBufOut.setpixel( 0,size-1,0, corner2 );

        imageBufOut.save();
    }

    for ( int i = 0; i < 6; i++) {
        delete resized[i];
    }

}


void Cubemap::fixupCubeEdges( const std::string& output, int level ) {

    fixImage( *this, level, output );
}
