/*
  median_cut.cpp by Tobias Alexander Franke (tob@cyberhead.de) 2013
  See http://www.tobias-franke.eu/?dev

  To get this to run, you will need stb_image.c and stb_image_write.h:
  http://nothings.org/stb_image.c
  http://nothings.org/stb/stb_image_write.h

  BSD License (http://www.opensource.org/licenses/bsd-license.php)

  Copyright (c) 2013, Tobias Alexander Franke (tob@cyberhead.de)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>

#define OPENIMAGE_LOADSAVE 1

#ifdef OPENIMAGE_LOADSAVE

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filter.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

OIIO_NAMESPACE_USING

#else

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.c"
#include "stb/stb_image_write.h"

#endif

struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
};

struct light
{
    float2 position;
    float area;
    float variance;

    // sort facility
    bool operator< (const light &rhs) const
    {
        return area >= rhs.area;
    }
};

template<typename T>
float luminance(T r, T g, T b)
{
    return r*0.2125f + g*0.7154f + b*0.0721f;
}

/**
 * Summed Area Table
 * 
 * Create a luminance summed area table from an image.
 */
class summed_area_table
{
protected:
    int width_, height_;
    std::vector<float> sat_;

    float I(int x, int y) const
    {
        if (x < 0 || y < 0) return 0;
        uint i = y*width_ + x;
        return sat_[i];
    }

public:
    template<typename T>
    void create_lum(T* rgb, uint width, uint height, uint nc)
    {
        assert(nc > 2);

        width_ = width; height_ = height;

        sat_.clear();
        sat_.resize(width_ * height_);

        for (uint y = 0; y < height_; ++y)
            for (uint x = 0; x < width_;  ++x)
            {
                uint i = y*width_ + x;

                T r = rgb[i*nc + 0];
                T g = rgb[i*nc + 1];
                T b = rgb[i*nc + 2];

                float ixy = luminance(r,g,b);

                sat_[i] = ixy + I(x-1, y) + I(x, y-1) - I(x-1, y-1);
            }
    }

    uint width() const  { return width_;  }
    uint height() const { return height_; }

    /**
     * Returns the sum of a region defined by A,B,C,D.
     *
     * A----B
     * |    |  sum = C+A-B-D
     * D----C
     */
    int sum(int ax, int ay, int bx, int by, int cx, int cy, int dx, int dy) const
    {
        return I(cx, cy) + I(ax, ay) - I(bx, by) - I(dx, dy);
    }
};

/**
 * A subregion in a summed_area_table.
 */
struct sat_region
{
    int x_, y_;
    uint w_, h_;
    float sum_;
    const summed_area_table* sat_;

    void create(int x, int y, uint w, uint h, const summed_area_table* sat, float init_sum = -1)
    {
        x_ = x; y_ = y; w_ = w; h_ = h; sum_ = init_sum; sat_ = sat;

        if (sum_ < 0)
            sum_ = sat_->sum(x,       y,
                             x+(w-1), y,
                             x+(w-1), y+(h-1),
                             x,       y+(h-1));
    }

    void split_w(sat_region& A) const
    {
        for (uint  w = 1; w <= w_; ++w)
        {
            A.create(x_, y_, w, h_, sat_);

            // if region left has approximately half the energy of the entire thing stahp
            if (A.sum_*2.f >= sum_)
                break;
        }
    }

    /**
     * Split region horizontally into subregions A and B.
     */
    void split_w(sat_region& A, sat_region& B) const
    {
        split_w(A);
        B.create(x_ + (A.w_-1), y_, w_ - A.w_, h_, sat_, sum_ - A.sum_);
    }

    void split_h(sat_region& A) const
    {
        for (size_t h = 1; h <= h_; ++h)
        {
            A.create(x_, y_, w_, h, sat_);

            // if region top has approximately half the energy of the entire thing stahp
            if (A.sum_*2.f >= sum_)
                break;
        }
    }

    /**
     * Split region vertically into subregions A and B.
     */
    void split_h(sat_region& A, sat_region& B) const
    {
        split_h(A);
        B.create(x_, y_ + (A.h_-1), w_, h_ - A.h_, sat_, sum_ - A.sum_);
    }

    float2 centroid() const
    {
        float2 c;

        sat_region A;

        split_w(A);
        c.x = A.x_ + (A.w_-1);

        split_h(A);
        c.y = A.y_ + (A.h_-1);

        return c;
    }
    
    float areaSize() const
    {
        return w_ * h_;
    }

    
};

/**
 * Recursively split a region r and append new subregions 
 * A and B to regions vector when at an end.
 */
void split_recursive(const sat_region& r, size_t n, std::vector<sat_region>& regions)
{
    // check: can't split any further?
    if (r.w_ < 2 || r.h_ < 2 || n == 0)
    {
        regions.push_back(r);
        return;
    }

    sat_region A, B;

    if (r.w_ > r.h_)
        r.split_w(A, B);
    else
        r.split_h(A, B);

    split_recursive(A, n-1, regions);
    split_recursive(B, n-1, regions);
}

/**
 * The median cut algorithm.
 *
 * img - Summed area table of an image
 * n - number of subdivision, yields 2^n cuts
 * regions - an empty vector that gets filled with generated regions
 */
void median_cut(const summed_area_table& img, size_t n, std::vector<sat_region>& regions)
{
    regions.clear();

    // insert entire image as start region
    sat_region r;
    r.create(0, 0, img.width(), img.height(), &img);

    // recursively split into subregions
    split_recursive(r, n, regions);
}

/**
 * Create a light source from each region by querying its centroid
 */
void red(float* d, int ci, uint m, uint nc)
{
    if (ci < 0) return;
    if (ci > m) return;

    d[ci + 0] = 1.f;
    d[ci + 1] = 0.f;
    if (nc > 1) d[ci + 2] = 0.f;
    if (nc > 2) d[ci + 3] = 1.f;
}

/**
 * Draw a cross at position l into image rgba
 */
void draw(float* rgba, uint width, uint height, light l, uint nc)
{
    static int i = 0;

    int ci;

    uint m = width*height*nc;

    // use l area/2 ?
    for (int x = -1; x < 2; ++x)
    {
        ci = std::min<int>((l.position.y*width + l.position.x+x)*nc, m);
        red(rgba, ci, m, nc);
    }

    ci = std::min<int>(((l.position.y+1)*width + l.position.x)*nc, m);
    red(rgba, ci, m, nc);

    ci = std::min<int>(((l.position.y-1)*width + l.position.x)*nc, m);
    red(rgba, ci, m, nc);
}

/**
 * Create a light source position from each region by querying its centroid
 */
void create_lights(const std::vector<sat_region>& regions, std::vector<light>& lights)
{

 
    for (std::vector<sat_region>::const_iterator r = regions.begin(); r != regions.end(); ++r)
    {
        std::cout << "Area : " << r->areaSize() << std::endl;

        light l;
        // set light at centroid
        l.position = r->centroid();
        // light area Size
        l.area = r->areaSize();
        //l.variance =
        lights.push_back(l);
    }

    // sort light by area Size
    // TODO: seek to get variance instead
    std::sort(lights.begin(), lights.end());

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Use " << argv[0] << " filename" << std::endl;
        return 1;
    }

    // load image
    int width, height, nc;
float *rgba;

#ifdef OPENIMAGE_LOADSAVE

    ImageInput* input = ImageInput::open ( argv[1] );

    const ImageSpec &spec (input->spec());
    width = spec.width;
    height = spec.height;
    nc = spec.nchannels;

    rgba = new  float[width*height*nc];
    input->read_image( TypeDesc::FLOAT, rgba);
    input->close();


#else

    rgba = stbi_loadf(argv[1], &width, &height, &nc, 0);
    if (stbi_failure_reason())
    {
        std::cerr << "stbi: " << stbi_failure_reason() << std::endl;
        return 1;
    }

#endif


    // create summed area table of luminance image
    summed_area_table lum_sat;
    lum_sat.create_lum(rgba, width, height, nc);

    // apply median cut
    std::vector<sat_region> regions;
    median_cut(lum_sat, 9, regions); // max 2^n cuts

    // create 2d positions from regions
    std::vector<light> lights;
    create_lights(regions, lights);

    // draw a marker into image for each position
    size_t i = 0;
    size_t lightNum = lights.size();

    // lightNum = 3;

    std::cout << "{ Lights: [";
    for (std::vector<light>::iterator l = lights.begin(); l != lights.end() && i < lightNum; ++l)
    {
        std::cout << "{ x: " << l->position.x << ", y: " << l->position.y << ", Area: " << l->area << "}" << std::endl;
        if (i < lightNum){

            std::cout << ",";

        }

        i++;

        //DEBUG ONLY
        draw(rgba, width, height, *l, nc);
    }
    std::cout << "]}";

    // save image with marked samples
    std::vector<unsigned char> conv;
    conv.resize(width*height*nc);

    for (i = 0; i < width * height * nc; ++i){
        conv[i] = static_cast<unsigned char>(rgba[i]*255);
    }

#ifdef OPENIMAGE_LOADSAVE

    ImageOutput* out = ImageOutput::create ("out/test_median_cut.png");
    ImageSpec specOut( width, height, nc, TypeDesc::UINT8);
    ImageOutput::OpenMode appendmode = ImageOutput::Create;
    out->open ("out/debug_median_cut.png", specOut, appendmode);
    out->write_image (TypeDesc::UINT8, &conv[0]);

#else
    stbi_write_bmp("debug_mediancut.bmp", width, height, nc, &conv[0]);
    
#endif

    return 0;
}
