/* Copyright (c) 2010-2013 Robert Kooima                                      */
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

#include <tiffio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <stdio.h>
#include <math.h>

#include "gray.h"
#include "sRGB.h"

/*----------------------------------------------------------------------------*/

/* In image structure represents an input or output raster.                   */

struct image
{
    float *p;  // data
    int    h;  // height
    int    w;  // width
    int    c;  // sample count
    int    b;  // sample depth
    int    s;  // sample format
};

typedef struct image image;

/* A pattern structure represents a supersampling pattern.                    */

struct point
{
    float i;
    float j;
};

typedef struct point point;

struct pattern
{
    int    n;
    point *p;
};

typedef struct pattern pattern;

/*----------------------------------------------------------------------------*/

typedef void (*filter)(const image *, float, float, float *);

typedef int (*to_img)(int *, float *, float *, int, int, const float *);
typedef int (*to_env)(int,   float,   float,   int, int,       float *);

/*----------------------------------------------------------------------------*/
/* A small set of single precision mathematical utilities.                    */

#define PI2 1.5707963f
#define PI  3.1415927f
#define TAU 6.2831853f

static inline float lerp(float a, float b, float k)
{
    return a * (1.f - k) + b * k;
}

static inline float clamp(float f, float a, float z)
{
    if      (f < a) return a;
    else if (f > z) return z;
    else            return f;
}

static inline float length(float i, float j)
{
    return sqrtf(i * i + j * j);
}

static inline void normalize(float *v)
{
    const float k = 1.f / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] *= k;
    v[1] *= k;
    v[2] *= k;
}

static inline void add(float *a, const float *b, const float *c)
{
    a[0] = b[0] + c[0];
    a[1] = b[1] + c[1];
    a[2] = b[2] + c[2];
}

/*----------------------------------------------------------------------------*/
/* Pixel block and line copying procedures used for cube map bordering.       */

static void blit(void *dst, int dw, int dx, int dy,
                 void *src, int sw, int sx, int sy,
                        int w, int h, int c, int b)
{
    int i;
    for (i = 0; i < h; i++)
        memcpy((char *) dst + ((i + dy) * dw + dx) * c * b,
               (char *) src + ((i + sy) * sw + sx) * c * b,
                                                 w * c * b);
}

#if 0
static void line(void *dst, int dw, int dx, int dy, int ddx, int ddy,
                 void *src, int sw, int sx, int sy, int sdx, int sdy,
                                                 int n, int c, int b)
{
    int i;
    for (i = 0; i < n; i++)
        memcpy((char *) dst + ((dy + ddy * i) * dw + dx + ddx * i) * c * b,
               (char *) src + ((sy + sdy * i) * sw + sx + sdx * i) * c * b,
                                                                     c * b);
}
#endif
/*----------------------------------------------------------------------------*/

/* Read one scanline from the given TIFF file, converting it from the format */
/* of the TIFF to float. The file must have contiguous planar configuration.  */
/* This is a drop-in replacement for TIFFReadScanline.                        */

int TIFFReadFloatScanline(TIFF *T, float *dst, uint32 r)
{
    static tdata_t *src = NULL;
    static tsize_t  len = 0;

    uint32 w = 0;
    uint16 c = 0;
    uint16 b = 0;
    uint16 s = 0;
    uint16 p = 0;

    TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &w);
    TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &c);
    TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &b);
    TIFFGetField(T, TIFFTAG_SAMPLEFORMAT,    &s);
    TIFFGetField(T, TIFFTAG_PLANARCONFIG,    &p);

    if (p == PLANARCONFIG_CONTIG)
    {
        if (len != TIFFScanlineSize(T))
        {
            len  = TIFFScanlineSize(T);
            src  = realloc(src, len);
        }

        if (src && TIFFReadScanline(T, src, r, 0) > 0)
        {
            if      ((b ==  8) && (s == SAMPLEFORMAT_UINT || s == 0))
                for (uint32 i = 0; i < w * c; i++)
                    dst[i] = ((uint8  *) src)[i] / 255.0f;

            else if ((b ==  8) && (s == SAMPLEFORMAT_INT))
                for (uint32 i = 0; i < w * c; i++)
                    dst[i] = ((int8   *) src)[i] / 127.0f;

            else if ((b == 16) && (s == SAMPLEFORMAT_UINT || s == 0))
                for (uint32 i = 0; i < w * c; i++)
                    dst[i] = ((uint16 *) src)[i] / 65535.0f;

            else if ((b == 16) && (s == SAMPLEFORMAT_INT))
                for (uint32 i = 0; i < w * c; i++)
                    dst[i] = ((int16  *) src)[i] / 32767.0f;

            else if ((b == 32) && (s == SAMPLEFORMAT_IEEEFP))
                for (uint32 i = 0; i < w * c; i++)
                    dst[i] = ((float  *) src)[i];

            else return -1;
        }
    }
    return +1;
}

/* Write one scanline to the given TIFF file, converting it from float to the */
/* format of the TIFF. The file must have contiguous planar configuration.    */
/* This is a drop-in replacement for TIFFWriteScanline.                       */

int TIFFWriteFloatScanline(TIFF *T, float *src, uint32 r)
{
    static tdata_t *dst = NULL;
    static tsize_t  len = 0;

    uint32 w = 0;
    uint16 c = 0;
    uint16 b = 0;
    uint16 s = 0;
    uint16 p = 0;

    TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &w);
    TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &c);
    TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &b);
    TIFFGetField(T, TIFFTAG_SAMPLEFORMAT,    &s);
    TIFFGetField(T, TIFFTAG_PLANARCONFIG,    &p);

    if (p == PLANARCONFIG_CONTIG)
    {
        if (len != TIFFScanlineSize(T))
        {
            len  = TIFFScanlineSize(T);
            dst  = realloc(dst, len);
        }

        if (dst)
        {
            if      ((b ==  8) && (s == SAMPLEFORMAT_UINT || s == 0))
                for (uint32 i = 0; i < w * c; i++)
                    ((uint8  *) dst)[i] = clamp(src[i], 0.0f, 1.0f) * 255.0f;

            else if ((b ==  8) && (s == SAMPLEFORMAT_INT))
                for (uint32 i = 0; i < w * c; i++)
                    ((int8   *) dst)[i] = clamp(src[i], 0.0f, 1.0f) * 127.0f;

            else if ((b == 16) && (s == SAMPLEFORMAT_UINT || s == 0))
                for (uint32 i = 0; i < w * c; i++)
                    ((uint16 *) dst)[i] = clamp(src[i], 0.0f, 1.0f) * 65535.0f;

            else if ((b == 16) && (s == SAMPLEFORMAT_INT))
                for (uint32 i = 0; i < w * c; i++)
                    ((int16  *) dst)[i] = clamp(src[i], 0.0f, 1.0f) * 32767.0f;

            else if ((b == 32) && (s == SAMPLEFORMAT_IEEEFP))
                for (uint32 i = 0; i < w * c; i++)
                    ((float  *) dst)[i] = src[i];

            else return -1;

            if (TIFFWriteScanline(T, dst, r, 0) == -1)
                return -1;
        }
    }
    return +1;
}

/*----------------------------------------------------------------------------*/

/* Allocate and initialize n image structures, each with a floating point     */
/* pixel buffer with width w, height h, and channel count c.                  */

static image *image_alloc(int n, int h, int w, int c, int b, int s)
{
    image *img;
    int f;

    if (s == 0)
    {
        if (b == 32)
            s = SAMPLEFORMAT_IEEEFP;
        else
            s = SAMPLEFORMAT_UINT;
    }

    if ((img = (image *) calloc(n, sizeof (image))))
        for (f = 0; f < n; f++)
        {
            img[f].p = (float *) calloc(w * h * c, sizeof (float));
            img[f].w = w;
            img[f].h = h;
            img[f].c = c;
            img[f].b = b;
            img[f].s = s;
        }

    return img;
}

/* Release the storage for n image buffers.                                   */
#if 0
static void image_free(image *img, int n)
{
    int f;

    for (f = 0; f < n; f++)
        free(img[f].p);

    free(img);
}
#endif

/* Read and return n pages from the named TIFF image file.                    */

static image *image_reader(const char *name, int n)
{
    image *in = 0;
    TIFF  *T  = 0;
    int    f;

    if ((T = TIFFOpen(name, "r")))
    {
        if ((in = (image *) calloc(n, sizeof (image))))
        {
            for (f = 0; f < n; f++)
            {
                uint16 b, c, s = 0;
                uint32 w, h, r;
                float *p;

                TIFFSetDirectory(T, f);

                TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &w);
                TIFFGetField(T, TIFFTAG_IMAGELENGTH,     &h);
                TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &c);
                TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &b);
                TIFFGetField(T, TIFFTAG_SAMPLEFORMAT,    &s);

                if ((p = (float *) malloc(h * w * c * sizeof (float))))
                {
                    for (r = 0; r < h; ++r)
                        TIFFReadFloatScanline(T, p + w * c * r, r);

                    in[f].p = (float *) p;
                    in[f].w = (int)     w;
                    in[f].h = (int)     h;
                    in[f].c = (int)     c;
                    in[f].b = (int)     b;
                    in[f].s = (int)     s;
                }
            }
        }
        TIFFClose(T);
    }
    return in;
}

/* Write n pages to the named TIFF image file.                                */

static void image_writer(const char *name, image *out, int n)
{
    TIFF  *T = 0;
    uint32 f, r;

    if ((T = TIFFOpen(name, "w")))
    {
        for (f = 0; f < n; ++f)
        {
            TIFFSetField(T, TIFFTAG_IMAGEWIDTH,      out[f].w);
            TIFFSetField(T, TIFFTAG_IMAGELENGTH,     out[f].h);
            TIFFSetField(T, TIFFTAG_SAMPLESPERPIXEL, out[f].c);
            TIFFSetField(T, TIFFTAG_BITSPERSAMPLE,   out[f].b);
            TIFFSetField(T, TIFFTAG_SAMPLEFORMAT,    out[f].s);
            TIFFSetField(T, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
            TIFFSetField(T, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

            if (out[f].c == 1)
            {
                TIFFSetField(T, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
                TIFFSetField(T, TIFFTAG_ICCPROFILE, sizeof (grayicc), grayicc);
            }
            else
            {
                TIFFSetField(T, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_RGB);
                TIFFSetField(T, TIFFTAG_ICCPROFILE, sizeof (sRGBicc), sRGBicc);
            }

            for (r = 0; r < out[f].h; ++r)
                TIFFWriteFloatScanline(T, out[f].p + out[f].w * out[f].c * r, r);

            TIFFWriteDirectory(T);
        }
        TIFFClose(T);
    }
}

/*----------------------------------------------------------------------------*/

#define SAMP(img, i, j, k) img.p[img.c * (img.w * i + j) + k]

static float *rotN(image *img, int i, int j)
{
    const int ii = i;
    const int jj = j;
    return img->p + img->c * (img->w * ii + jj);
}

static float *rotL(image *img, int i, int j)
{
    const int ii = j;
    const int jj = img->h - i - 1;
    return img->p + img->c * (img->w * ii + jj);
}

static float *rotR(image *img, int i, int j)
{
    const int ii = img->w - j - 1;
    const int jj = i;
    return img->p + img->c * (img->w * ii + jj);
}

typedef float *(*rot)(image *, int, int);

static void border(image *a, rot rota, image *b, rot rotb, int d)
{
    const size_t s = b->c * sizeof (float);
    const int    n = b->h;

    for     (int i = d; i < n - d; i++)
        for (int j = 0; j <     d; j++)
        {
            memcpy(rota(a, i, n - d + j), rotb(b, i,         d + j), s);
            memcpy(rotb(b, i,         j), rota(a, i, n - d - d + j), s);
        }
}

/* Add borders to a cubemap image. Assume the given image pointer is an array */
/* of six images. Copy each to a new sef of six images, each two pixels wider */
/* and higher. Also copy the borders. This is necessary for correct cubemap   */
/* sampling.                                                                  */

static image *image_border(image *src)
{
    image *dst = 0;

    const int d = 1;

    if ((src) && (dst = image_alloc(6, src[0].w + 2 * d,
                                       src[0].w + 2 * d,
                                       src[0].c,
                                       src[0].b,
                                       src[0].s)))
    {
        const int n = src[0].w;
        const int c = src[0].c;
        const int b = 4;

        const int N = n + 2 * d;

        /* Copy all page data. */

        blit(dst[0].p, N, d, d, src[0].p, n, 0, 0, n, n, c, b);
        blit(dst[1].p, N, d, d, src[1].p, n, 0, 0, n, n, c, b);
        blit(dst[2].p, N, d, d, src[2].p, n, 0, 0, n, n, c, b);
        blit(dst[3].p, N, d, d, src[3].p, n, 0, 0, n, n, c, b);
        blit(dst[4].p, N, d, d, src[4].p, n, 0, 0, n, n, c, b);
        blit(dst[5].p, N, d, d, src[5].p, n, 0, 0, n, n, c, b);

        border(dst + 0, rotN, dst + 5, rotN, d);
        border(dst + 5, rotN, dst + 1, rotN, d);
        border(dst + 1, rotN, dst + 4, rotN, d);
        border(dst + 4, rotN, dst + 0, rotN, d);

        border(dst + 1, rotR, dst + 2, rotN, d);
        border(dst + 1, rotL, dst + 3, rotN, d);

        border(dst + 2, rotN, dst + 0, rotL, d);
        border(dst + 3, rotN, dst + 0, rotR, d);

        border(dst + 2, rotL, dst + 4, rotL, d);
        border(dst + 2, rotR, dst + 5, rotL, d);
        border(dst + 3, rotL, dst + 5, rotR, d);
        border(dst + 3, rotR, dst + 4, rotR, d);

#if 0
        /* Corner patch hack. */

        for     (f = 0; f < 6; f++)
            for (k = 0; k < c; k++)
            {
                SAMP(dst[f], 0, 0, k) = (SAMP(dst[f], 1, 0, k) +
                                         SAMP(dst[f], 0, 1, k) +
                                         SAMP(dst[f], 1, 1, k)) / 3.0f;
                SAMP(dst[f], 0, M, k) = (SAMP(dst[f], 1, M, k) +
                                         SAMP(dst[f], 0, L, k) +
                                         SAMP(dst[f], 1, L, k)) / 3.0f;
                SAMP(dst[f], M, 0, k) = (SAMP(dst[f], L, 0, k) +
                                         SAMP(dst[f], M, 1, k) +
                                         SAMP(dst[f], L, 1, k)) / 3.0f;
                SAMP(dst[f], M, M, k) = (SAMP(dst[f], L, M, k) +
                                         SAMP(dst[f], M, L, k) +
                                         SAMP(dst[f], L, L, k)) / 3.0f;
            }
#endif
    }
    return dst;
}

/*----------------------------------------------------------------------------*/

/* Sample an image at row i column j using linear interpolation.              */

static void filter_linear(const image *img, float i, float j, float *p)
{
    const float ii = clamp(i - 0.5f, 0.0f, img->h - 1.0f);
    const float jj = clamp(j - 0.5f, 0.0f, img->w - 1.0f);

    const long  i0 = lrintf(floorf(ii)), i1 = lrintf(ceilf(ii));
    const long  j0 = lrintf(floorf(jj)), j1 = lrintf(ceilf(jj));

    const float di = ii - i0;
    const float dj = jj - j0;

    int k;

    for (k = 0; k < img->c; k++)
        p[k] += lerp(lerp(img->p[(img->w * i0 + j0) * img->c + k],
                          img->p[(img->w * i0 + j1) * img->c + k], dj),
                     lerp(img->p[(img->w * i1 + j0) * img->c + k],
                          img->p[(img->w * i1 + j1) * img->c + k], dj), di);
}

/* Sample an image at row i column j using nearest neighbor.                  */

static void filter_nearest(const image *img, float i, float j, float *p)
{
    const float ii = clamp(i - 0.5f, 0.0f, img->h - 1.0f);
    const float jj = clamp(j - 0.5f, 0.0f, img->w - 1.0f);

    const long  i0 = lrintf(ii);
    const long  j0 = lrintf(jj);

    int k;

    for (k = 0; k < img->c; k++)
        p[k] += img->p[(img->w * i0 + j0) * img->c + k];
}

/*----------------------------------------------------------------------------*/

static int cube_to_img(int *f, float *i, float *j, int h, int w, const float *v)
{
    const float X = fabsf(v[0]);
    const float Y = fabsf(v[1]);
    const float Z = fabsf(v[2]);

    float x;
    float y;

    if      (v[0] > 0 && X >= Y && X >= Z) { *f = 0; x = -v[2] / X; y = -v[1] / X; }
    else if (v[0] < 0 && X >= Y && X >= Z) { *f = 1; x =  v[2] / X; y = -v[1] / X; }
    else if (v[1] > 0 && Y >= X && Y >= Z) { *f = 2; x =  v[0] / Y; y =  v[2] / Y; }
    else if (v[1] < 0 && Y >= X && Y >= Z) { *f = 3; x =  v[0] / Y; y = -v[2] / Y; }
    else if (v[2] > 0 && Z >= X && Z >= Y) { *f = 4; x =  v[0] / Z; y = -v[1] / Z; }
    else if (v[2] < 0 && Z >= X && Z >= Y) { *f = 5; x = -v[0] / Z; y = -v[1] / Z; }
    else return 0;

    *i = 1.0f + (h - 2) * (y + 1.0f) / 2.0f;
    *j = 1.0f + (w - 2) * (x + 1.0f) / 2.0f;

    return 1;
}

static int dome_to_img(int *f, float *i, float *j, int h, int w, const float *v)
{
    if (v[1] >= 0)
    {
        const float d = sqrtf(v[0] * v[0] + v[2] * v[2]);
        const float r = acosf(v[1]) / PI2;

        *f = 0;
        *i = h * (1.0f - r * v[2] / d) / 2.0f;
        *j = w * (1.0f + r * v[0] / d) / 2.0f;

        return 1;
    }
    return 0;
}

static int hemi_to_img(int *f, float *i, float *j, int h, int w, const float *v)
{
    if (v[2] <= 0)
    {
        const float d = sqrtf(v[0] * v[0] + v[1] * v[1]);
        const float r = acosf(-v[2]) / PI2;

        *f = 0;
        *i = h * (1.0f - r * v[1] / d) / 2.0f;
        *j = w * (1.0f + r * v[0] / d) / 2.0f;

        return 1;
    }
    return 0;
}

static int ball_to_img(int *f, float *i, float *j, int h, int w, const float *v)
{
    const float d = sqrtf(v[0] * v[0] + v[1] * v[1]);
    const float r = sinf(acosf(v[2]) * 0.5f);

    *f = 0;
    *i = h * (1.0f - r * v[1] / d) / 2.0f;
    *j = w * (1.0f + r * v[0] / d) / 2.0f;

    return 1;
}

static int rect_to_img(int *f, float *i, float *j, int h, int w, const float *v)
{
    *f = 0;
    *i = h * (       acosf (v[1])        / PI);
    *j = w * (0.5f + atan2f(v[0], -v[2]) / TAU);

    return 1;
}

/*----------------------------------------------------------------------------*/

static int cube_to_env(int f, float i, float j, int h, int w, float *v)
{
    const int p[6][3][3] = {
        {{  0,  0, -1 }, {  0, -1,  0 }, {  1,  0,  0 }},
        {{  0,  0,  1 }, {  0, -1,  0 }, { -1,  0,  0 }},
        {{  1,  0,  0 }, {  0,  0,  1 }, {  0,  1,  0 }},
        {{  1,  0,  0 }, {  0,  0, -1 }, {  0, -1,  0 }},
        {{  1,  0,  0 }, {  0, -1,  0 }, {  0,  0,  1 }},
        {{ -1,  0,  0 }, {  0, -1,  0 }, {  0,  0, -1 }},
    };

    const float y = 2.0f * i / h - 1.0f;
    const float x = 2.0f * j / w - 1.0f;

    v[0] = p[f][0][0] * x + p[f][1][0] * y + p[f][2][0];
    v[1] = p[f][0][1] * x + p[f][1][1] * y + p[f][2][1];
    v[2] = p[f][0][2] * x + p[f][1][2] * y + p[f][2][2];

    normalize(v);
    return 1;
}

static int dome_to_env(int f, float i, float j, int h, int w, float *v)
{
    const float y = 2.0f * i / h - 1.0f;
    const float x = 2.0f * j / w - 1.0f;

    if (length(x, y) <= 1.0f)
    {
        const float lat = PI2 - PI2 * length(x, y);
        const float lon =             atan2f(x, y);

        v[0] =  sinf(lon) * cosf(lat);
        v[1] =              sinf(lat);
        v[2] = -cosf(lon) * cosf(lat);

        return 1;
    }
    return 0;
}

static int hemi_to_env(int f, float i, float j, int h, int w, float *v)
{
    const float y = 2.0f * i / h  - 1.0f;
    const float x = 2.0f * j / w  - 1.0f;

    if (length(x, y) <= 1.0f)
    {
        const float lat = PI2 - PI2 * length(x, y);
        const float lon =             atan2f(x, y);

        v[0] =  sinf(lon) * cosf(lat);
        v[1] = -cosf(lon) * cosf(lat);
        v[2] =             -sinf(lat);

        return 1;
    }
    return 0;
}

static int ball_to_env(int f, float i, float j, int h, int w, float *v)
{
    const float y = 2.0f * i / h  - 1.0f;
    const float x = 2.0f * j / w  - 1.0f;

    if (length(x, y) <= 1.0f)
    {
        const float lat = 2.0f * asin(length(x, y));
        const float lon =             atan2f(x, y);

        v[0] =  sinf(lon) * sinf(lat);
        v[1] = -cosf(lon) * sinf(lat);
        v[2] =              cosf(lat);

        return 1;
    }
    return 0;
}

static int rect_to_env(int f, float i, float j, int h, int w, float *v)
{
    const float lat = PI2 - PI * i / h;
    const float lon = TAU * j / w - PI;

    v[0] =  sinf(lon) * cosf(lat);
    v[1] =              sinf(lat);
    v[2] = -cosf(lon) * cosf(lat);

    return 1;
}

/*----------------------------------------------------------------------------*/

static int xfm(const float *rot, float *v)
{
    if (rot[0])
    {
        float s = sinf(rot[0] * PI / 180.0f);
        float c = cosf(rot[0] * PI / 180.0f);
        float y = v[1] * c - v[2] * s;
        float z = v[1] * s + v[2] * c;

        v[1] = y; v[2] = z;
    }
    if (rot[1])
    {
        float s = sinf(rot[1] * PI / 180.0f);
        float c = cosf(rot[1] * PI / 180.0f);
        float z = v[2] * c - v[0] * s;
        float x = v[2] * s + v[0] * c;

        v[0] = x; v[2] = z;
    }
    if (rot[2])
    {
        float s = sinf(rot[2] * PI / 180.0f);
        float c = cosf(rot[2] * PI / 180.0f);
        float x = v[0] * c - v[1] * s;
        float y = v[0] * s + v[1] * c;

        v[0] = x; v[1] = y;
    }
    return 1;
}

void supersample(const image   *src,
                 const image   *dst,
                 const pattern *pat,
                 const float   *rot,
                 filter fil, to_img img, to_env env, int f, int i, int j)
{
    int    F;
    float  I;
    float  J;
    int    k;
    int    c = 0;
    float *p = dst[f].p + dst[f].c * (dst[f].w * i + j);

    /* For each sample of the supersampling pattern... */

    for (k = 0; k < pat->n; k++)
    {
        const float ii = pat->p[k].i + i;
        const float jj = pat->p[k].j + j;

        /* Project and unproject giving the source location. Sample there. */

        float v[3];

        if (env( f, ii, jj, dst->h, dst->w, v) && xfm(rot, v) &&
            img(&F, &I, &J, src->h, src->w, v))
        {
#if 1
            fil(src + F, I, J, p);
#else
            p[0] = (v[0] + 1.0f) / 2.0f;
            p[1] = (v[1] + 1.0f) / 2.0f;
            p[2] = (v[2] + 1.0f) / 2.0f;
#endif
            c++;
        }
    }

    /* Normalize the sample. */

    for (k = 0; k < dst->c; k++)
        p[k] /= c;
}

void process(const image   *src,
             const image   *dst,
             const pattern *pat,
             const float   *rot,
             filter fil, to_img img, to_env env, int n)
{
    int i;
    int j;
    int f;

    /* Sample all destination rows, columns, and pages. */

    #pragma omp parallel for private(j, f)
    for         (i = 0; i < dst->h; i++)
        for     (j = 0; j < dst->w; j++)
            for (f = 0; f <      n; f++)
                supersample(src, dst, pat, rot, fil, img, env, f, i, j);
}

/*----------------------------------------------------------------------------*/

static point cent_points[] = {
    { 0.5f, 0.5f },
};

static point rgss_points[] = {
    { 0.125f, 0.625f },
    { 0.375f, 0.125f },
    { 0.625f, 0.875f },
    { 0.875f, 0.375f },
};

static point box2_points[] = {
    { 0.25f, 0.25f },
    { 0.25f, 0.75f },
    { 0.75f, 0.25f },
    { 0.75f, 0.75f },
};

static point box3_points[] = {
    { 0.1666667f, 0.1666667f },
    { 0.1666667f, 0.5000000f },
    { 0.1666667f, 0.8333333f },
    { 0.5000000f, 0.1666667f },
    { 0.5000000f, 0.5000000f },
    { 0.5000000f, 0.8333333f },
    { 0.8333333f, 0.1666667f },
    { 0.8333333f, 0.5000000f },
    { 0.8333333f, 0.8333333f },
};

static point box4_points[] = {
    { 0.125f, 0.125f },
    { 0.125f, 0.375f },
    { 0.125f, 0.625f },
    { 0.125f, 0.875f },
    { 0.375f, 0.125f },
    { 0.375f, 0.375f },
    { 0.375f, 0.625f },
    { 0.375f, 0.875f },
    { 0.625f, 0.125f },
    { 0.625f, 0.375f },
    { 0.625f, 0.625f },
    { 0.625f, 0.875f },
    { 0.875f, 0.125f },
    { 0.875f, 0.375f },
    { 0.875f, 0.625f },
    { 0.875f, 0.875f },
};

static const pattern cent_pattern = {  1, cent_points };
static const pattern rgss_pattern = {  4, rgss_points };
static const pattern box2_pattern = {  4, box2_points };
static const pattern box3_pattern = {  9, box3_points };
static const pattern box4_pattern = { 16, box4_points };

/*----------------------------------------------------------------------------*/

static int usage(const char *exe)
{
    fprintf(stderr,
            "%s [-i input] [-o output] [-p pattern] [-f filter] [-n n] src dst\n"
            "\t-i ... Input  file type: cube, dome, hemi, ball, rect  [rect]\n"
            "\t-o ... Output file type: cube, dome, hemi, ball, rect  [rect]\n"
            "\t-p ... Sample pattern: cent, rgss, box2, box3, box4    [rgss]\n"
            "\t-f ... Filter type: nearest, linear                  [linear]\n"
            "\t-n ... Output size                                     [1024]\n",
            exe);
    return 0;
}

int main(int argc, char **argv)
{
    /* Set some default behaviors. */

    const char *i = "rect";
    const char *o = "rect";
    const char *p = "rgss";
    const char *f = "linear";

    float rot[3] = { 0.f, 0.f, 0.f };

    int n = 1024;
    int c;

    /* Parse the command line options. */

    while ((c = getopt(argc, argv, "i:o:p:n:f:x:y:z:")) != -1)
        switch (c)
        {
            case 'i': i      = optarg;               break;
            case 'o': o      = optarg;               break;
            case 'p': p      = optarg;               break;
            case 'f': f      = optarg;               break;
            case 'x': rot[0] = strtod(optarg, 0);    break;
            case 'y': rot[1] = strtod(optarg, 0);    break;
            case 'z': rot[2] = strtod(optarg, 0);    break;
            case 'n': n      = strtol(optarg, 0, 0); break;

            default: return usage(argv[0]);
        }

    int      num = 1;
    image   *src = 0;
    image   *dst = 0;
    image   *tmp = 0;
    to_img   img;
    to_env   env;
    filter   fil;

    /* Select the sampler. */

    if      (!strcmp(f, "linear"))  fil = filter_linear;
    else if (!strcmp(f, "nearest")) fil = filter_nearest;
    else return usage(argv[0]);

    /* Read the input image. */

    if (optind + 2 <= argc)
    {
        if      (!strcmp(i, "cube"))
        {
            tmp = image_reader(argv[optind], 6);
            src = image_border(tmp);
            img = cube_to_img;
        }
        else if (!strcmp(i, "dome"))
        {
            src = image_reader(argv[optind], 1);
            img = dome_to_img;
        }
        else if (!strcmp(i, "hemi"))
        {
            src = image_reader(argv[optind], 1);
            img = hemi_to_img;
        }
        else if (!strcmp(i, "ball"))
        {
            src = image_reader(argv[optind], 1);
            img = ball_to_img;
        }
        else if (!strcmp(i, "rect"))
        {
            src = image_reader(argv[optind], 1);
            img = rect_to_img;
        }
        else return usage(argv[0]);
    }
    else return usage(argv[0]);

    /* Prepare the output image. */

    if (src)
    {
        if      (!strcmp(o, "cube"))
        {
            dst = image_alloc((num = 6), n, n, src->c, src->b, src->s);
            env = cube_to_env;
        }
        else if (!strcmp(o, "dome"))
        {
            dst = image_alloc((num = 1), n, n, src->c, src->b, src->s);
            env = dome_to_env;
        }
        else if (!strcmp(o, "hemi"))
        {
            dst = image_alloc((num = 1), n, n, src->c, src->b, src->s);
            env = hemi_to_env;
        }
        else if (!strcmp(o, "ball"))
        {
            dst = image_alloc((num = 1), n, n, src->c, src->b, src->s);
            env = ball_to_env;
        }
        else if (!strcmp(o, "rect"))
        {
            dst = image_alloc((num = 1), n, 2 * n, src->c, src->b, src->s);
            env = rect_to_env;
        }
        else return usage(argv[0]);
    }

    /* Perform the remapping using the selected pattern. */

    if (src && dst)
    {
        if      (!strcmp(p, "cent"))
            process(src, dst, &cent_pattern, rot, fil, img, env, num);

        else if (!strcmp(p, "rgss"))
            process(src, dst, &rgss_pattern, rot, fil, img, env, num);

        else if (!strcmp(p, "box2"))
            process(src, dst, &box2_pattern, rot, fil, img, env, num);

        else if (!strcmp(p, "box3"))
            process(src, dst, &box3_pattern, rot, fil, img, env, num);

        else if (!strcmp(p, "box4"))
            process(src, dst, &box4_pattern, rot, fil, img, env, num);

        else return usage(argv[0]);

        /* Write the output. */

        image_writer(argv[optind + 1], dst, num);
    }

    return 0;
}
