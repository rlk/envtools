/*
  median_cut.cpp by Tobias Alexander Franke (tob@cyberhead.de) 2013
  See http://www.tobias-franke.eu/?dev
  BSD License (http://www.opensource.org/licenses/bsd-license.php)
  Copyright (c) 2013, Tobias Alexander Franke (tob@cyberhead.de)
*/


#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <float.h>


#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filter.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

OIIO_NAMESPACE_USING

#include "Math"
#include "ExtractLightsVariance"
#include "SummedAreaTable"
#include "SummedAreaTableRegion"

/**
 * Recursively split a region r and append new subregions
 * A and B to regions vector when at an end.
 */
void splitRecursive(const satRegion& r, const uint n, std::vector<satRegion>& regions)
{
    // check: can't split any further?
    if (r._w < 2 || r._h < 2 || n == 0)
    {
        // only now add region
        regions.push_back(r);
        return;
    }

    satRegion A, B;

    if (r._w > r._h)
        r.split_w(A, B);
    else
        r.split_h(A, B);

    if (A._h > 2 && A._w > 2 ) {
        splitRecursive(A, n-1, regions);
    }

    if (B._h > 2 && B._w > 2 ) {
        splitRecursive(B, n-1, regions);
    }

}

/**
 * The median cut algorithm Or Variance Minimisation
 *
 * img - Summed area table of an image
 * n - number of subdivision, yields 2^n cuts
 * regions - an empty vector that gets filled with generated regions
 */
void medianVarianceCut(const SummedAreaTable& img, const uint n, std::vector<satRegion>& regions)
{
    regions.clear();

    // insert entire image as start region
    satRegion r;
    r.create(0, 0, img.width(), img.height(), &img);

    // recursively split into subregions
    splitRecursive(r, n, regions);
}

uint mergeLights(std::vector<light>& lights, std::vector<light>& newLights, uint width, uint height)
{

    // AreaSize under which we merge
    const uint mergeindexPos =  (lights.size() * 25) / 100;
    const double mergeAreaSize = lights[mergeindexPos]._areaSize;

    const uint border = 5;
    uint numMergedLightTotal = 0;

    for (std::vector<light>::iterator lightIt = lights.begin(); lightIt != lights.end(); ++lightIt)
    {
        // already merged, we do nothing
        if (lightIt->_merged) continue;


        std::vector<light>::iterator lCurrent = lightIt;

        uint x = lCurrent->_x;
        uint y = lCurrent->_y;
        uint w = lCurrent->_w;
        uint h = lCurrent->_h;

        uint numMergedLight;
        do{

            numMergedLight = 0;

            for (std::vector<light>::iterator l = lights.begin(); l != lights.end(); ++l) {

                // ignore already merged and itself
                if (l->_merged || l == lCurrent || l->_mergedNum > 0 ) continue;

                // ignore too big
                if (mergeAreaSize < l->_areaSize) continue;

                bool intersect2D = !(l->_y-border > y+h || l->_y+l->_h+border < y || l->_x-border > x + w || l->_x+l->_w+border < x);
                // try left/right border as it's a env wrap
                // complexity arise, how to merge...and then retest after
                /*
                  if (!intersect2D ){
                  if( x == 0 ){
                  //check left borders
                  intersect2D = !(l->_y-border > y+h || l->_y+l->_h+border < y || l->_x-border > width + w || l->_x+l->_w+border < width);
                  }else if( x+w == width ){
                  //check right borders
                  intersect2D = !(l->_y-border > y+h || l->_y+l->_h+border < y || l->_x-border > w + (width - x) || l->_x+l->_w+border < (width - x));
                  }
                  }
                */

                //  share borders
                if (intersect2D) {

                    // goes after next merged
                    l->_merged = true;

                    lCurrent->_x = std::min(x, l->_x);
                    lCurrent->_y = std::min(y, l->_y);

                    lCurrent->_w = std::max(x + w, l->_x + l->_w) - lCurrent->_x;
                    lCurrent->_h = std::max(y + h, l->_y + l->_h) - lCurrent->_y;

                    x = lCurrent->_x;
                    y = lCurrent->_y;
                    w = lCurrent->_w;
                    h = lCurrent->_h;

                    // light is bigger, better candidate to main light
                    lCurrent->_mergedNum++;
                    lCurrent->_sum += l->_sum;

                    numMergedLight++;
                }
            }

        } while (numMergedLight > 0);

        if (lCurrent->_mergedNum > 0){

            lCurrent->_areaSize = lCurrent->_w * lCurrent->_h;
            lCurrent->_lumAverage = lCurrent->_sum / lCurrent->_areaSize;

            //lCurrent->_sortCriteria = lCurrent->_lumAverage;
            lCurrent->_sortCriteria = lCurrent->_sum;

            newLights.push_back(*lCurrent);

            numMergedLightTotal += lCurrent->_mergedNum;
        }

    }

    for (std::vector<light>::iterator lCurrent = lights.begin(); lCurrent != lights.end(); ++lCurrent)
    {
        // add remaining lights
        if (!lCurrent->_merged && lCurrent->_mergedNum == 0){

            lCurrent->_lumAverage = lCurrent->_sum / lCurrent->_areaSize;

            //lCurrent->_sortCriteria = lCurrent->_lumAverage;
            lCurrent->_sortCriteria = lCurrent->_sum;

            newLights.push_back(*lCurrent);
        }
    }

    return numMergedLightTotal;

}


/**
 * Create a light source position from each region by querying its centroid
 * Merge small area light neighbouring
 */
void createAndMergeLights(const std::vector<satRegion>& regions, std::vector<light>& lights, std::vector<light>& newLights, float *rgba, const uint width, const uint height, const uint nc)
{

    // convert region into lights
    for (std::vector<satRegion>::const_iterator region = regions.begin(); region != regions.end(); ++region)
    {

        light l;

        // init values
        l._merged = false;
        l._mergedNum = 0;

        l._x = region->_x;
        l._y = region->_y;
        l._w = region->_w;
        l._h = region->_h;

        // set light at centroid
        l._centroidPosition = region->centroid();
        // light area Size
        l._areaSize = region->areaSize();

        // sat lum sum of area
        l._sum = region->getSum();

        // sat lum sum of area
        l._variance = region->getVariance();

        // average Result
        l._lumAverage = region->getMean();
        l._rAverage = region->_r / l._areaSize;
        l._gAverage = region->_g / l._areaSize;
        l._bAverage = region->_b / l._areaSize;


        l._sortCriteria = l._areaSize;
        //l._sortCriteria = l._lumAverage;

        const uint i = static_cast<uint>(l._centroidPosition._y*width + l._centroidPosition._x);

        double r = rgba[i*nc + 0];
        double g = rgba[i*nc + 1];
        double b = rgba[i*nc + 2];
        l._luminancePixel = luminance(r,g,b);

        lights.push_back(l);
    }

    // sort light
    std::sort(lights.begin(), lights.end());

    #define MERGE 1
    #ifdef MERGE

    uint mergedLights = mergeLights(lights, newLights, width, height);
    // sort By sum now (changed the sortCriteria during merge)
    std::sort(newLights.begin(), newLights.end());
    std::reverse(newLights.begin(), newLights.end());

#else
    newLights.resize(lights.size());
    std::copy(lights.begin(), lights.end(), newLights.begin());
#endif

}


void outputJSON(const std::vector<light> &lights, uint height, uint width, uint imageAreaSize)
{
    size_t i = 0;
    size_t lightNum = lights.size();

    lightNum = 15;


    std::cout << "[";

    for (std::vector<light>::const_iterator l = lights.begin(); l != lights.end() && i < lightNum; ++l) {

        // convert x,y to direction
        double3 d;

        //https://www.shadertoy.com/view/4dsGD2
        double theta = (1.0 - l->_centroidPosition._y / height) * PI;
        double phi   = l->_centroidPosition._x / width * TAU;
	
        // Equation from http://graphicscodex.com  [sphry]
	    d._x =  sin(theta) * sin(phi);
        d._y =             cos(theta);
        d._z =  sin(theta) * cos(phi);
        
        // normalize direction
        const double norm = sqrtf( d._x*d._x + d._y*d._y + d._z*d._z );
        if (norm < 1e-16) {
            const double inv = 1.0f/norm;
            d._x *= inv;
            d._y *= inv;
            d._z *= inv;
        }

        // convert to bitmap
        const int rCol = static_cast<uchar>(static_cast<uchar>(l->_rAverage*255));
        const int gCol = static_cast<uchar>(static_cast<uchar>(l->_gAverage*255));
        const int bCol = static_cast<uchar>(static_cast<uchar>(l->_bAverage*255));

        // 1 JSON object per light
        std::cout << "{";
        std::cout << " \"position\": [" << l->_centroidPosition._x << ", " << l->_centroidPosition._y << "] ,";
        std::cout << " \"direction\": [" << d._x << ", " << d._y << ", " << d._z << "], ";
        std::cout << " \"luminosity\": " << l->_lumAverage << ", ";
        std::cout << " \"color\": [" << rCol << ", " << gCol << ", " << bCol << "], ";
        std::cout << " \"size\": " << l->_areaSize/imageAreaSize << ", ";
        std::cout << " \"area\": {\"x\":" << l->_x << ", \"y\":" << l->_y << ", \"w\":" << l->_w << ", \"h\":" << l->_h << "}, ";
        std::cout << " \"variance\": " << l->_variance << " ";
        std::cout << " }" << std::endl;

        if (i < lightNum - 1){

            std::cout << ",";

        }

        i++;
    }

    std::cout << "]";

}


////////////////////////////////////////////////
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Use " << argv[0] << " filename" << std::endl;
        return 1;
    }

    ////////////////////////////////////////////////
    // load image
    int width, height, nc;
    float *rgba;

    ImageInput* input = ImageInput::open ( argv[1] );
    const ImageSpec &spec (input->spec());
    width = spec.width;
    height = spec.height;
    nc = spec.nchannels;
    const uint imageAreaSize = width*height;
    rgba = new float[imageAreaSize*nc];
    input->read_image( TypeDesc::FLOAT, rgba);
    input->close();

    ////////////////////////////////////////////////
    // create summed area table of luminance image
    SummedAreaTable lum_sat;

    lum_sat.createLum(rgba, width, height, nc);

    ////////////////////////////////////////////////
    // apply cut alogrithm
    std::vector<satRegion> regions;

    medianVarianceCut(lum_sat, 8, regions); // max 2^n cuts

    ////////////////////////////////////////////////
    // create Lights from regions
    std::vector<light> lights;
    std::vector<light> mainLights;

    createAndMergeLights(regions, lights, mainLights, rgba, width, height, nc);

    ////////////////////////////////////////////////
    // output JSON

    // do we want to output/save original same variance light ?
    // Merged Light sorted By Area Size
    // outputJSON(lights);

    // Merged Light sorted By Luminance intensity
    outputJSON(mainLights, height, width, imageAreaSize);


    return 0;
}
