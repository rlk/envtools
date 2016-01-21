#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define PI 3.14159265358979f
#define PI_INV (1.0f/PI)



static constant float4 CubemapFace[6][3] = {
    { {0.0f, 0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 0.0f}, { 1.0f, 0.0f, 0.0f, 0.0f} },// x positif
    { {0.0f, 0.0f,  1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f, 0.0f} }, // x negatif

    { {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f,  1.0f, 0.0f}, {0.0f,  1.0f, 0.0f, 0.0f} },  // y positif
    { {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 0.0f} }, // y negatif

    { {1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 0.0f}, {0.0f, 0.0f,  1.0f, 0.0f} },  // z positif
    {{-1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f, 0.0f} } // z negatif
};


static float2 hammersley(uint i, uint N) {
    uint bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);

    double u = (double)(i) / (double)(N);
    double v = (double)(bits);
    v = v * 2.3283064365386963e-10;

    return (float2)((float)(u), (float)(v));
}

static double D_GGX( float NdotH, float alpha)
{
    // use GGX / Trowbridge-Reitz, same as Disney and Unreal 4
    // cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
    double tmp = alpha / (NdotH*NdotH*(alpha*alpha-1.0)+1.0);
    return tmp * tmp * PI_INV;
}

constant float3 ViewVector = (float3)(0,0,1);
kernel void computeLightVector( global write_only float4* precomputedLightVector,
                                global write_only float* nol,
                                volatile global read_write int* currentLightVectorIndex,
                                const uint numValidSamples,
                                const uint numSamples,
                                const uint size,
                                const float roughnessLinear)
{
    uint i = get_global_id(0);

    float2 Xi = hammersley( i, numSamples );
    float4 result = { 0.0f, 0.0f, 0.0f, 0.0f};

    float roughness = roughnessLinear * roughnessLinear;
    float Phi = 2.0f * PI * Xi[0];
    float CosTheta = sqrt( (1.0f - Xi[1]) / ( 1.0f + ( roughness * roughness - 1.0f) * Xi[1] ) );
    float SinTheta = sqrt( 1.0f - CosTheta * CosTheta );
    float3 H;
    H[0] = SinTheta * cos( Phi );
    H[1] = SinTheta * sin( Phi );
    H[2] = CosTheta;
    H = normalize(H);

    float3 L =  ( dot( ViewVector, H ) * 2.0f ) * H - ViewVector;
    L = normalize(L);

    float NoL = L[2];
    if ( NoL <= 1e-5f ) return;

    // H[2] is NoH in local space
    // adds 1.e-5 to avoid D_GGX / 0.0
    float NoH = H[2] + 1.e-5f;

    // data to evaluate pdf
    float VoH = NoH;

    // Probability Distribution Function
    float Pdf = D_GGX(NoH, roughnessLinear) * NoH / (4.0f * VoH);

    // Solid angle represented by this sample
    float omegaS = 1.0f / (numValidSamples * Pdf);

    // Solid angle covered by 1 pixel with 6 faces that are EnvMapSize X EnvMapSize
    float omegaP = 4.0f * PI / ( 6.0f * size * size);

    // Original paper suggest biasing the mip to improve the results
    // log2 seems broken on osx with intel gpu need to use native_log2
    float mipBias = 1.0f; // I tested that the result is better with bias 1
    float mipLevel = max(0.5f * native_log2(omegaS / omegaP) + mipBias, 0.0f);
    // printf("omegaP %f omegaS %f div %f miplevel %f\n", omegaP, omegaS, native_log2(omegaS / omegaP), mipLevel);

    result[0] = L[0];
    result[1] = L[1];
    result[2] = L[2];
    result[3] = mipLevel;

    uint index = atom_inc(&currentLightVectorIndex[0]);
    if ( index >= numValidSamples)
        printf("problem");
    precomputedLightVector[index] = result;
    nol[index] = NoL;
}



kernel void computeTapVector( global __write_only float4* tapVector,
                              global __write_only float* weight,
                              const float sigmaSqr,
                              const float radius,
                              const uint numSamples)
{
    uint i = get_global_id(0);

    float2 Xi = hammersley( i, numSamples );

    float u = Xi[0];
    float v = Xi[1];
    float angle = u * PI * 2.0f;

    // uniform
    float r = sqrt( v ) * radius;

    // not uniform
    //float r = v * radius;

    float x = r * cos(angle);
    float y = r * sin(angle);

    // compute gaussian weight
    // https://en.wikipedia.org/wiki/Gaussian_blur
    // http://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size
    // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch40.html
    //float standardDeviation = 0.84089642;
    //float sigmaSqr = sigma*sigma;
    // weight = exp(-(x*x + y*y)/twoStandardDeviationSqr)/( PI * twoStandardDeviationSqr );
    float w = exp(-0.5f*(x*x + y*y)/sigmaSqr);

    float4 H;
    H[0] = x;
    H[1] = y;
    H[2] = 1.0f;
    H[3] = w;
    H = normalize(H);

    tapVector[i] = H;
    weight[i] = w;
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

static float4 vectToTexelCoord(const float4 direction) {

    uint bestAxis = 0;
    if ( fabs(direction[1]) > fabs(direction[0]) ) {
        bestAxis = 1;
        if ( fabs(direction[2]) > fabs(direction[1]) )
            bestAxis = 2;
    } else if ( fabs(direction[2]) > fabs(direction[0]) )
        bestAxis = 2;

    // select the index of cubemap face
    uint faceIndex = bestAxis * 2 + ( direction[bestAxis] > 0.0f ? 0 : 1 );
    float bestAxisValue = direction[bestAxis];
    float denom = fabs( bestAxisValue );

    //float maInv = 1.0/denom;
    float4 dir = direction / denom;

    float4 f0 = CubemapFace[faceIndex][0];
    float4 f1 = CubemapFace[faceIndex][1];
    float sc = dot(f0, dir);
    float tc = dot(f1, dir);
    float ppx = (sc + 1.0f) * 0.5f;
    float ppy = (tc + 1.0f) * 0.5f;

    // printf ("dir %f %f %f\n", dir[0], dir[1], dir[2] );
    return (float4)( ppx, ppy, (float)(faceIndex), (float)(faceIndex) );
}

static float4 texelCoordToVect(uint face, float ui, float vi, uint sizeImage, bool fixup) {

    float u,v;
    float size = (float)(sizeImage);
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
    float4 axis0 = CubemapFace[face][0];
    float4 axis1 = CubemapFace[face][1];
    float4 vecZ = CubemapFace[face][2];

    float4 vecX = axis0 * u;
    float4 vecY = axis1 * v;
    float4 result = normalize(  vecX + vecY + vecZ );
    // printf ("axis0 %f %f %f #axis1 %f %f %f #axis2 %f %f %f # %f %f face %d\n", axis0[0], axis0[1], axis0[2], axis1[0], axis1[1], axis1[2], axis2[0], axis2[1], axis2[2], u, v, face);

    return result;
}

constant sampler_t cubemapSampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// getSample is generated by python
GET_SAMPLE_IMPLEMENTATION

// float4 getSample( float level, float4 uv, read_only image2d_array_t cubemap0 ,read_only image2d_array_t cubemap1,read_only image2d_array_t cubemap2,read_only image2d_array_t cubemap3,read_only image2d_array_t cubemap4,read_only image2d_array_t cubemap5,read_only image2d_array_t cubemap6,read_only image2d_array_t cubemap7,read_only image2d_array_t cubemap8,read_only image2d_array_t cubemap9,read_only image2d_array_t cubemap10 )
// {
//     float r;
//     float4 color0,color1;

//     //return read_imagef( cubemap1, cubemapSampler, uv );

//     if (level < 1.0f ) {
//         r = level;
//         color0 = read_imagef( cubemap0, cubemapSampler, uv );
//         color1 = read_imagef( cubemap1, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 2.0f ) {
//         r = level-1.0f;
//         color0 = read_imagef( cubemap1, cubemapSampler, uv );
//         color1 = read_imagef( cubemap2, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 3.0f ) {
//         r = level-2.0f;
//         color0 = read_imagef( cubemap2, cubemapSampler, uv );
//         color1 = read_imagef( cubemap3, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 4.0f ) {
//         r = level-3.0f;
//         color0 = read_imagef( cubemap3, cubemapSampler, uv );
//         color1 = read_imagef( cubemap4, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 5.0f ) {
//         r = level-4.0f;
//         color0 = read_imagef( cubemap4, cubemapSampler, uv );
//         color1 = read_imagef( cubemap5, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 6.0f ) {
//         r = level-5.0f;
//         color0 = read_imagef( cubemap5, cubemapSampler, uv );
//         color1 = read_imagef( cubemap6, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 7.0f ) {
//         r = level-6.0f;
//         color0 = read_imagef( cubemap6, cubemapSampler, uv );
//         color1 = read_imagef( cubemap7, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 8.0f ) {
//         r = level-7.0f;
//         color0 = read_imagef( cubemap7, cubemapSampler, uv );
//         color1 = read_imagef( cubemap8, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 9.0f ) {
//         r = level-8.0f;
//         color0 = read_imagef( cubemap8, cubemapSampler, uv );
//         color1 = read_imagef( cubemap9, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     if (level < 10.0f ) {
//         r = level-9.0f;
//         color0 = read_imagef( cubemap9, cubemapSampler, uv );
//         color1 = read_imagef( cubemap10, cubemapSampler, uv );
//         return mix(color0, color1, r);
//     }
//     return read_imagef( cubemap10, cubemapSampler, uv );
// }



kernel void computeGGX( uint face,
                        write_only image2d_t faceResult,
                        global read_only float4* precomputedLightVector,
                        const float totalWeight,
                        const uint nbSamples,
                        const uint fixup,
                        MIPMAP_LEVEL_ARGUMENTS )
//                          read_only image2d_array_t cubemap0, ... )
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    uint size = get_image_width(faceResult);

    float4 N = texelCoordToVect( face, (float)(i), (float)(j), size, (bool)fixup );
    float4 prefilteredColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f );

    float4 UpVector = fabs(N[2]) < 0.999 ? (float4)(0,0,1,0) : (float4)(1,0,0,0);
    float4 TangentX = normalize( cross( UpVector, N ) );
    float4 TangentY = normalize( cross( N, TangentX ) );

    float4 color, uv, L;
    float4 LworldSpace;
    float NoL;
    float lod = 0.0;
    for( uint n = 0; n < nbSamples; n++ ) {

        // vec4 contains the light vector + miplevel
        L = precomputedLightVector[ n ];
        // printf ("dir %f %f %f\n", N[0], N[1], N[2] );

        NoL = L[2];
        lod = L[3];

        LworldSpace = TangentX * L[0] + TangentY * L[1] + N * L[2];

        //LworldSpace = N;
        // get uv, face
        uv = vectToTexelCoord( LworldSpace );

        //printf("UV %f %f %f %f # %f %f %f # %d %d\n", uv[0], uv[1], uv[2], uv[3], N[0], N[1], N[2], i, j);
        // color = read_imagef( cubemap0, cubemapSampler, uv );
        color = GET_SAMPLE_CALL
        prefilteredColor += color * NoL;
    }

    prefilteredColor = prefilteredColor / totalWeight;
    // prefilteredColor = read_imagef( cubemap0, cubemapSampler, uv); // (float4)(i*1.0/1024,j*1.0/1024, (float)(face), (float)(face) ));
    write_imagef( faceResult, (int2)(i,j), prefilteredColor );
}



kernel void computeBackground( uint face,
                               write_only image2d_t faceResult,
                               global read_only float4* tapVector,
                               const float totalWeight,
                               const uint nbSamples,
                               const uint fixup,
                               read_only image2d_array_t cubemap0)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    uint size = get_image_width(faceResult);

    float4 N = texelCoordToVect( face, (float)(i), (float)(j), size, (bool)fixup );
    float4 finalColor = (float4)(0.0f, 0.0f, 0.0f, 0.0f );

    float4 UpVector = fabs(N[2]) < 0.999 ? (float4)(0,0,1,0) : (float4)(1,0,0,0);
    float4 TangentX = normalize( cross( UpVector, N ) );
    float4 TangentY = normalize( cross( N, TangentX ) );

    float4 color, uv, L;
    float4 LworldSpace;
    for( uint n = 0; n < nbSamples; n++ ) {

        L = tapVector[ n ];
        LworldSpace = TangentX * L[0] + TangentY * L[1] + N * L[2];

        // get uv, face
        uv = vectToTexelCoord( LworldSpace );

        color = read_imagef( cubemap0, cubemapSampler, uv );
        finalColor += color * L[3];
    }

    finalColor = finalColor / totalWeight;
    write_imagef( faceResult, (int2)(i,j), finalColor );
}
