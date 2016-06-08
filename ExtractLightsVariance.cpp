/*
  median_cut.cpp by Tobias Alexander Franke (tob@cyberhead.de) 2013
  See http://www.tobias-franke.eu/?dev
  BSD License (http://www.opensource.org/licenses/bsd-license.php)
  Copyright (c) 2013, Tobias Alexander Franke (tob@cyberhead.de)
*/


#include <iostream>
#include <getopt.h>
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
#include "ExtractLightsMerge"
#if !defined(NDEBUG)
#include "ExtractLightsVarianceDebug"
#endif

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



void outputJSON(const std::vector<light> &lights, uint height, uint width, uint imageAreaSize, double luminanceSum)
{
    size_t i = 0;
    size_t lightNum = lights.size();

    std::cout << "[";
    
    double globalVariance = luminanceSum /imageAreaSize;
    
    for (std::vector<light>::const_iterator l = lights.begin(); l != lights.end() && i < lightNum; ++l) {

        const double x = l->_centroidPosition._y / height;        
        const double y = l->_centroidPosition._x / width;
        
        const double w = static_cast <double>(l->_w) / width;        
        const double h = static_cast <double>(l->_h) / height;
        
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

        // convert to float
        const float rCol = l->_rAverage;        
        const float gCol = l->_gAverage;
        const float bCol = l->_bAverage;


        // 1 JSON object per light
        std::cout << "{";
        std::cout << " \"position\": [" << x << ", " << y << "] ,";
        std::cout << " \"direction\": [" << d._x << ", " << d._y << ", " << d._z << "], ";
        std::cout << " \"luminosity\": " << (l->_lumAverage) << ", ";
        std::cout << " \"color\": [" << rCol << ", " << gCol << ", " << bCol << "], ";
        std::cout << " \"area\": {\"x\":" << x << ", \"y\":" << y << ", \"w\":" << w << ", \"h\":" << h << "}, ";
        std::cout << " \"sum\": " << (l->_sum ) << ", ";
        std::cout << " \"variance\": " << (l->_variance ) << " ";
        std::cout << " }" << std::endl;

        if (i < lightNum - 1){

            std::cout << ",";

        }

        i++;
    }

    std::cout << "]";

}

////////////////////////////////////////////////
static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [-a Areasize] [-r ratioLight] [-n numCuts] file.hdr" << std::endl;
    return 1;
}

////////////////////////////////////////////////
int main(int argc, char** argv)
{
    // max area encased by light extracted, ratio of env map size
    // default is using Area of 5% of EnvMap as dir approx light
    float ratioAreaSizeMax = 0.01f;
    // Idea is to limit light extraction to analytic directional light
    //  So we must limite area and power extracted
    // as more the power means more difference with Env lighting
    // when we compute shadow as
    // "real time shadows =  lightEnv - LightExtracted"
    // ratioLight = luminanceLight / luminanceEnv
    float ratioLuminanceLight = 0.5f;// ratio of lightExtracted On Global Illumination sum
    int numCuts = 8;// number of division squared of the envmap of same lighting power

    int c;    
    while ((c = getopt(argc, argv, "a:r:n")) != -1)
    {        
        switch (c)
        {
        case 'a': ratioAreaSizeMax = atof(optarg); break;
        case 'r': ratioLuminanceLight = atof(optarg); break;
        case 'n': numCuts = atoi(optarg); break;

        default: return usage(argv[0]);
        }
    }
    

    if ( optind < argc )
    {
        ////////////////////////////////////////////////
        // load image
        int width, height, nc;
        float *rgba;

        ImageInput* input = ImageInput::open ( argv[optind] );
    
        if (!input) {
            std::cerr << "Cannot open " << argv[1] << " image file" << std::endl;
            return 1;
        }
    
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
        // apply cut algorithm
        std::vector<satRegion> regions;

        medianVarianceCut(lum_sat, numCuts, regions); // max 2^n cuts

        if (regions.empty())
        {
            std::cerr << "Cannot cut " << argv[1] << " into light regions" << std::endl;
            return 1;            
        }
        
        ////////////////////////////////////////////////
        // create Lights from regions
        std::vector<light> lights;
        std::vector<light> mainLights;

        //
        // convert absolute input parameters
        // to relative to environment at hand value.
        // From ratio to pixel squared area
        /// Light Max luminance in percentage 
        double luminanceSum = lum_sat.sum(0,0,
                                          width-1,0,
                                          width-1,height-1,
                                          0,height-1);
        
        const double luminanceMaxLight = ratioLuminanceLight*luminanceSum;

        // And he saw that light was good, and separated light from darkness
        createLightsFromRegions(regions, lights, rgba, width, height, nc);
        
        // sort lights
        // the smaller, the more powerful luminance
        std::sort(lights.begin(), lights.end());
    
        

#define MERGE 1
#ifdef MERGE
          
        // Light Area Size under which we merge
        // default to size of the median region size
        // if lots of small lights => give small area
        // if lots of big lights => give big area
        const uint mergeindexPos =  (lights.size() * 25) / 100;
        const double mergeAreaSize = lights[mergeindexPos]._areaSize;
        //const double mergeAreaSize = 0.1 * imageAreaSize;
        
        uint mergedLights = mergeLights(lights, mainLights, width, height, mergeAreaSize, luminanceMaxLight);

        
        // sort By sum now (changed the sort Criteria during merge)
        // biggest Sum first
        std::sort(mainLights.begin(), mainLights.end());
        std::reverse(mainLights.begin(), mainLights.end());

#define SELECT 1
#ifdef SELECT
        
        // now keep lights from inside merged Zone
        lights.clear();
        const double areaSizeMax = ratioAreaSizeMax * imageAreaSize;
        mergedLights = selectLights(mainLights, lights, width, height, areaSizeMax, luminanceMaxLight, luminanceSum);

        mainLights.clear();
        mainLights.resize(lights.size());
        std::copy(lights.begin(), lights.end(), mainLights.begin());
        
        // sort By sum now (changed the sort Criteria during merge)
        // biggest sum first
        std::sort(mainLights.begin(), mainLights.end());
        std::reverse(mainLights.begin(), mainLights.end());
        
#endif // SELECT
        
#else
        
        newLights.resize(lights.size());
        std::copy(lights.begin(), lights.end(), newLights.begin());
        
#endif
        
        ////////////////////////////////////////////////
        // output JSON

        // do we want to output/save original same variance light ?
        // Merged Light sorted By Area Size
        //outputJSON(lights, height, width, imageAreaSize );

        // Merged Light sorted By Luminance intensity
        outputJSON(mainLights, height, width, imageAreaSize, luminanceSum);


#if !defined(NDEBUG)

        debugDrawLight(regions, lights, mainLights, rgba, width, height, nc);

#endif // !defined(NDEBUG)
        
    }
    else{
        
        return usage( argv[0] );
        
    }
    
    return 0;
}
