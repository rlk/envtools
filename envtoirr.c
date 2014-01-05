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
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <stdio.h>
#include <math.h>

#include "header.h"
#include "gray.h"
#include "sRGB.h"

/* See "An Efficient Representation for Irradiance Environment Maps" by       */
/* Ramamorrthi & Handrahan, SIGGRAPH 2001                                     */

/*----------------------------------------------------------------------------*/

/* This array defines the X, Y, and Z axes of each of the six cube map faces, */
/* which determine the orientation of each face and the mapping between 2D    */
/* image space and 3D cube map space.                                         */

static const double nx[3] = { -1.0,  0.0,  0.0 };
static const double px[3] = {  1.0,  0.0,  0.0 };
static const double ny[3] = {  0.0, -1.0,  0.0 };
static const double py[3] = {  0.0,  1.0,  0.0 };
static const double nz[3] = {  0.0,  0.0, -1.0 };
static const double pz[3] = {  0.0,  0.0,  1.0 };

const double *cubemap_axis[6][3] = {
    { pz, py, nx },
    { nz, py, px },
    { nx, nz, ny },
    { nx, pz, py },
    { nx, py, nz },
    { px, py, pz },
};

static void normalize(double *a, const double *b)
{
    const double k = 1.0 / sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);

    a[0] = b[0] * k;
    a[1] = b[1] * k;
    a[2] = b[2] * k;
}

void transform(double *a, const double *M, const double *b)
{
    a[0] = M[ 0] * b[0] + M[ 4] * b[1] + M[ 8] * b[2] + M[12] * b[3];
    a[1] = M[ 1] * b[0] + M[ 5] * b[1] + M[ 9] * b[2] + M[13] * b[3];
    a[2] = M[ 2] * b[0] + M[ 6] * b[1] + M[10] * b[2] + M[14] * b[3];
    a[3] = M[ 3] * b[0] + M[ 7] * b[1] + M[11] * b[2] + M[15] * b[3];
}

static void vector(double *v, int f, double x, double y, double z)
{
    const double *X = cubemap_axis[f][0];
    const double *Y = cubemap_axis[f][1];
    const double *Z = cubemap_axis[f][2];

    double w[3];

    w[0] = X[0] * x + Y[0] * y + Z[0] * z;
    w[1] = X[1] * x + Y[1] * y + Z[1] * z;
    w[2] = X[2] * x + Y[2] * y + Z[2] * z;

    normalize(v, w);
    v[3] = 1.0;
}

double solid_angle(const double *a,
                   const double *b,
                   const double *c)
{
    double n = fabs(a[0] * (b[1] * c[2] - b[2] * c[1]) +
                    a[1] * (b[2] * c[0] - b[0] * c[2]) +
                    a[2] * (b[0] * c[1] - b[1] * c[0]));

    double d = 1.0 + a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
                   + a[0] * c[0] + a[1] * c[1] + a[2] * c[2]
                   + b[0] * c[0] + b[1] * c[1] + b[2] * c[2];

    return 2.0 * atan2(n, d);
}

/*----------------------------------------------------------------------------*/

double *sph_grace_cathedral(uint16 *C)
{
    double *L = 0;

    if ((L = (double *) malloc(9 * 3 * sizeof (double))))
    {
        L[ 0] =  0.79; L[ 1] =  0.44; L[ 2] =  0.54;
        L[ 3] =  0.39; L[ 4] =  0.35; L[ 5] =  0.60;
        L[ 6] = -0.34; L[ 7] = -0.18; L[ 8] = -0.27;
        L[ 9] = -0.29; L[10] = -0.06; L[11] =  0.01;
        L[12] = -0.11; L[13] = -0.05; L[14] = -0.12;
        L[15] = -0.26; L[16] = -0.22; L[17] = -0.47;
        L[18] = -0.16; L[19] = -0.09; L[20] = -0.15;
        L[21] =  0.56; L[22] =  0.21; L[23] =  0.14;
        L[24] =  0.21; L[25] = -0.05; L[26] = -0.30;

        *C = 3;
    }
    return L;
}

double *sph_eucalyptus_grove(uint16 *C)
{
    double *L = 0;

    if ((L = (double *) malloc(9 * 3 * sizeof (double))))
    {
        L[ 0] =  0.38; L[ 1] =  0.43; L[ 2] =  0.45;
        L[ 3] =  0.29; L[ 4] =  0.36; L[ 5] =  0.41;
        L[ 6] =  0.04; L[ 7] =  0.03; L[ 8] =  0.01;
        L[ 9] = -0.10; L[10] = -0.10; L[11] = -0.09;
        L[12] = -0.06; L[13] = -0.06; L[14] = -0.04;
        L[15] =  0.01; L[16] = -0.01; L[17] = -0.05;
        L[18] = -0.09; L[19] = -0.13; L[20] = -0.15;
        L[21] = -0.06; L[22] = -0.05; L[23] = -0.04;
        L[24] =  0.02; L[25] = -0.00; L[26] = -0.05;

        *C = 3;
    }
    return L;
}

double *sph_st_peters_basilica(uint16 *C)
{
    double *L = 0;

    if ((L = (double *) malloc(9 * 3 * sizeof (double))))
    {
        L[ 0] =  0.36; L[ 1] =  0.26; L[ 2] =  0.23;
        L[ 3] =  0.18; L[ 4] =  0.14; L[ 5] =  0.13;
        L[ 6] = -0.02; L[ 7] = -0.01; L[ 8] = -0.00;
        L[ 9] =  0.03; L[10] =  0.02; L[11] =  0.01;
        L[12] =  0.02; L[13] =  0.01; L[14] =  0.00;
        L[15] = -0.05; L[16] = -0.03; L[17] = -0.01;
        L[18] = -0.09; L[19] = -0.08; L[20] = -0.07;
        L[21] =  0.01; L[22] =  0.00; L[23] =  0.00;
        L[24] = -0.08; L[25] = -0.06; L[26] =  0.00;

        *C = 3;
    }
    return L;
}

/*----------------------------------------------------------------------------*/

double calc_domega(const double *v00,
                   const double *v01,
                   const double *v10,
                   const double *v11)
{
    return (solid_angle(v00, v11, v01) +
            solid_angle(v11, v00, v10)) / (4 * PI);
}

void calc_Y(double *Y, double x, double y, double z)
{
    Y[0] = 0.282095;
    Y[1] = 0.488603 *  y;
    Y[2] = 0.488603 *  z;
    Y[3] = 0.488603 *  x;
    Y[4] = 1.092548 *  x * y;
    Y[5] = 1.092548 *  y * z;
    Y[6] = 0.315392 * (3 * z * z - 1);
    Y[7] = 1.092548 *  x * z;
    Y[8] = 0.546274 * (x * x - y * y);
}

void env_to_sph_face(double *L, TIFF *I, int F, uint32 N, uint16 C, uint16 B)
{
    /* Confirm the parameters of the current face. */

    if (TIFFSetDirectory(I, F))
    {
        uint32 w, h;
        uint16 b, c;
        float *s;

        TIFFGetField(I, TIFFTAG_IMAGEWIDTH,      &w);
        TIFFGetField(I, TIFFTAG_IMAGELENGTH,     &h);
        TIFFGetField(I, TIFFTAG_SAMPLESPERPIXEL, &c);
        TIFFGetField(I, TIFFTAG_BITSPERSAMPLE,   &b);

        if (w == N && h == N && b == B && c == C)
        {
            /* Allocate a scanline buffer. */

            if ((s = (float *) malloc(TIFFScanlineSize(I))))
            {
                uint32 i;
                uint32 j;
                uint16 k;

                /* Iterate over all pixels of the current cube face. */

                for (i = 0; i < N; ++i)
                {
                    if (TIFFReadScanline(I, s, i, 0) == 1)
                    {
                        for (j = 0; j < N; ++j)
                        {
                            float *p = s + j * c;

                            /* Compute the direction vector to this pixel. */

                            double y0  = (2.0 * (i      ) - N) / N;
                            double y   = (2.0 * (i + 0.5) - N) / N;
                            double y1  = (2.0 * (i + 1.0) - N) / N;
                            double x0  = (2.0 * (j      ) - N) / N;
                            double x   = (2.0 * (j + 0.5) - N) / N;
                            double x1  = (2.0 * (j + 1.0) - N) / N;

                            double v00[4];
                            double v01[4];
                            double v10[4];
                            double v11[4];

                            double v[4];
                            double Y[9];
                            double dd;

                            vector(v00, F, x0, y0, 1.f);
                            vector(v01, F, x0, y1, 1.f);
                            vector(v10, F, x1, y0, 1.f);
                            vector(v11, F, x1, y1, 1.f);
                            vector(v,   F, x,  y,  1.f);

                            dd = calc_domega(v00, v01, v10, v11);

                            calc_Y(Y, v[0], v[1], v[2]);

                            for (k = 0; k < c; ++k)
                            {
                                L[0 * c + k] += Y[0] * (double) p[k] * dd;
                                L[1 * c + k] += Y[1] * (double) p[k] * dd;
                                L[2 * c + k] += Y[2] * (double) p[k] * dd;
                                L[3 * c + k] += Y[3] * (double) p[k] * dd;
                                L[4 * c + k] += Y[4] * (double) p[k] * dd;
                                L[5 * c + k] += Y[5] * (double) p[k] * dd;
                                L[6 * c + k] += Y[6] * (double) p[k] * dd;
                                L[7 * c + k] += Y[7] * (double) p[k] * dd;
                                L[8 * c + k] += Y[8] * (double) p[k] * dd;
                            }
                        }
                    }
                }
                free(s);
            }
        }
    }
}

double *env_to_sph(TIFF *I, uint32 N, uint16 C, uint16 B)
{
    double *L = 0;
    int     F;

    /* Allocate storage for spherical harmonic coefficients. */

    if ((L = (double *) calloc(9 * C, sizeof (double))))
    {
        /* Iterate over all faces, accumulating coefficients for all. */

        for (F = 0; F < 6; ++F)
            env_to_sph_face(L, I, F, N, C, B);
    }
    return L;
}

/*----------------------------------------------------------------------------*/

void irrmatrix(double *M, double Lzz, double Lpn, double Lpz,
                          double Lpp, double Lqm, double Lqn,
                          double Lqz, double Lqp, double Lqq)
{
    const double c1 = 0.429043;
    const double c2 = 0.511664;
    const double c3 = 0.743125;
    const double c4 = 0.886227;
    const double c5 = 0.247708;

    M[ 0] =  c1 * Lqq;
    M[ 1] =  c1 * Lqm;
    M[ 2] =  c1 * Lqp;
    M[ 3] =  c2 * Lpp;

    M[ 4] =  c1 * Lqm;
    M[ 5] = -c1 * Lqq;
    M[ 6] =  c1 * Lqn;
    M[ 7] =  c2 * Lpn;

    M[ 8] =  c1 * Lqp;
    M[ 9] =  c1 * Lqn;
    M[10] =  c3 * Lqz;
    M[11] =  c2 * Lpz;

    M[12] =  c2 * Lpp;
    M[13] =  c2 * Lpn;
    M[14] =  c2 * Lpz;
    M[15] =  c4 * Lzz - c5 * Lqz;
}

void sph_to_irr_face(float *d, const double *M, int F, uint32 N, uint16 C, uint16 k)
{
    uint32 i;
    uint32 j;

    /* Iterate over all pixels of the current cube face. */

    for     (i = 0; i < N; ++i)
        for (j = 0; j < N; ++j)
        {
            /* Compute the direction vector of the current pixel. */

            double y = (double) (2.0 * (i + 0.5) - N) / N;
            double x = (double) (2.0 * (j + 0.5) - N) / N;

            double v[4];
            double w[4];

            vector(v, F, x, y, 1.0);

            /* Compute the irradiance. */

            transform(w, M, v);

            (d + i * N * C + j * C)[k] = (float) (v[0] * w[0] +
                                                  v[1] * w[1] +
                                                  v[2] * w[2] +
                                                  v[3] * w[3]);
        }
}

void sph_to_irr(double *L, TIFF *O, uint32 N, uint16 C)
{
    double M[16];
    float *d;
    uint16 k;
    uint32 i;
    size_t S;
    int    F;

    /* Allocate a scratch buffer for one cube face. */

    if ((d = (float *) malloc(N * N * C * sizeof (float))))
    {
        /* Iterate over all cube faces. */

        for (F = 0; F < 6; ++F)
        {
            TIFFSetField(O, TIFFTAG_IMAGEWIDTH,      N);
            TIFFSetField(O, TIFFTAG_IMAGELENGTH,     N);
            TIFFSetField(O, TIFFTAG_SAMPLESPERPIXEL, C);
            TIFFSetField(O, TIFFTAG_BITSPERSAMPLE,   32);
            TIFFSetField(O, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
            TIFFSetField(O, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
            TIFFSetField(O, TIFFTAG_SAMPLEFORMAT,    SAMPLEFORMAT_IEEEFP);

            if (C == 1)
            {
                TIFFSetField(O, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_MINISBLACK);
                TIFFSetField(O, TIFFTAG_ICCPROFILE, sizeof (grayicc), grayicc);
            }
            else
            {
                TIFFSetField(O, TIFFTAG_PHOTOMETRIC,  PHOTOMETRIC_RGB);
                TIFFSetField(O, TIFFTAG_ICCPROFILE, sizeof (sRGBicc), sRGBicc);
            }

            S = (size_t) TIFFScanlineSize(O);

            /* Iterate over all channels. */

            for (k = 0; k < C; ++k)
            {
                /* Generate the irradiance matrix for the current channel. */

                irrmatrix(M, L[0 * C + k],
                             L[1 * C + k],
                             L[2 * C + k],
                             L[3 * C + k],
                             L[4 * C + k],
                             L[5 * C + k],
                             L[6 * C + k],
                             L[7 * C + k],
                             L[8 * C + k]);

                /* Render the current cube face using the current irradiance. */

                sph_to_irr_face(d, M, F, N, C, k);
            }

            /* Write the result to an image file. */

            for (i = 0; i < N; ++i)
                TIFFWriteScanline(O, (uint8 *) d + S * i, i, 0);

            TIFFWriteDirectory(O);
        }
        free(d);
    }
}

/*----------------------------------------------------------------------------*/

static void sph_dump(const double *L, int c)
{
    int i;
    int k;

    printf("Spherical harmonic coefficients:\n");

    for (i = 0; i < 9; ++i)
    {
        for (k = 0; k < c; ++k)
            printf("% 12.5f ", L[i * c + k]);

        printf("\n");
    }
}

static double *sph_file(const char *name, uint16 *C)
{
    double *L = 0;
    TIFF   *T;

    if ((T = TIFFOpen(name, "r")))
    {
        uint32 W, H;
        uint16 B;

        TIFFGetField(T, TIFFTAG_IMAGEWIDTH,     &W);
        TIFFGetField(T, TIFFTAG_IMAGELENGTH,    &H);
        TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, C);
        TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,  &B);

        if (W == H)
            L = env_to_sph(T, W, *C, B);

        TIFFClose(T);
    }
    return L;
}

/*----------------------------------------------------------------------------*/

static int usage(const char *exe)
{
    fprintf(stderr, "Usage: %s [-n n] [-f in.tif] [-ges] out.tif n\n", exe);
    return 1;
}

int main(int argc, char *argv[])
{
    double *L = 0;
    TIFF   *T;
    uint16  C;
    int     n = 256;
    int     c;

    while ((c = getopt(argc, argv, "n:f:ges")) != -1)
        switch (c)
        {
            case 'n': n = strtol(optarg, 0, 0);       break;
            case 'f': L = sph_file(optarg, &C);       break;
            case 'g': L = sph_grace_cathedral   (&C); break;
            case 'e': L = sph_eucalyptus_grove  (&C); break;
            case 's': L = sph_st_peters_basilica(&C); break;

            default: return usage(argv[0]);
        }

    if (L && optind < argc)
    {
        if ((T = TIFFOpen(argv[optind], "w")))
        {
            sph_dump(L, C);
            sph_to_irr(L, T, n, C);
            TIFFClose(T);
        }
        free(L);
    }
    else return usage(argv[0]);

    return 0;
}
