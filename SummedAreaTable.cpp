#include <cassert>
#include "SummedAreaTable"

void SummedAreaTable::createLum(float* rgb, const uint width, const uint height, const uint nc)
{
    assert(nc > 2);

    _width = width;
    _height = height;

    const uint imgSize = width * height;

    _sat.clear();
    _sat.resize(imgSize);
    _sat1.clear();
    _sat1.resize(imgSize);
    _sat2.clear();
    _sat2.resize(imgSize);
    _sat3.clear();
    _sat3.resize(imgSize);
    _sat4.clear();
    _sat4.resize(imgSize);
    _sat5.clear();
    _sat5.resize(imgSize);

    _r.clear();
    _r.resize(imgSize);
    _g.clear();
    _g.resize(imgSize);
    _b.clear();
    _b.resize(imgSize);

    _sat[0] = 0.0;
    _sat1[0] = 0.0;
    _sat2[0] = 0.0;
    _sat3[0] = 0.0;
    _sat4[0] = 0.0;
    _sat5[0] = 0.0;
    _r[0] = 0.0;
    _g[0] = 0.0;
    _b[0] = 0.0;

    double weightAccum = 0.0;

    // solid angle for 1 pixel on equi map
    double weight = (4.0 * PI) / ((double)(imgSize));

    Vec3f d;
    uint faceIdx;
    float aU, aV;


    _minLum = DBL_MAX;
    _maxLum = DBL_MIN;
    _minPonderedLum = DBL_MAX;
    _maxPonderedLum = DBL_MIN;
    _minR = DBL_MAX;
    _maxR = DBL_MIN;
    _minB = DBL_MAX;
    _maxG = DBL_MIN;
    _minB = DBL_MAX;
    _maxB = DBL_MIN;
    _sum = 0.0;
    
    for (uint y = 0; y < height; ++y) {
        
        const double posY = (double)(y+1.0) / (double)(height+1.0);

        // the latitude-longitude format overrepresents the area of regions near the poles.
        // To compensate for this, the pixels of the probe image
        // should first be scaled by cosφ.
        // (φ == 0 at middle height of image input)
        const double solidAngle = cos(PI* (posY - 0.5)) * weight;

        for (uint x = 0; x < width;  ++x) {

            const uint i = y*width + x;

            double r = rgb[i*nc + 0];
            double g = rgb[i*nc + 1];
            double b = rgb[i*nc + 2];

            double ixy = luminance(r,g,b);

            // update Min/Max before pondering            
            _minLum = std::min(ixy, _minLum);
            _maxLum = std::max(ixy, _maxLum);
            
            _minR = std::min(r, _minR);
            _maxR = std::max(r, _maxR);
            
            _minG = std::min(g, _minG);
            _maxG = std::max(g, _maxG);
            
            _minB = std::min(b, _minB);
            _maxB = std::max(b, _maxB);

#define _PONDER_REAL 
#ifdef _PONDER_REAL

            
            r *= solidAngle* imgSize;
            g *= solidAngle* imgSize;
            b *= solidAngle* imgSize;
            
            // ixy = luminance(r,g,b);
            // pondering luminance for unpondered colors makes more sense
            ixy *= solidAngle;
            
#else
            // complex approx going through cubemap conversion
            
            // convert panorama to direction x,y,z
            //https://www.shadertoy.com/view/4dsGD2
            double theta = (1.0 - posY) * PI;
            double phi   = (double) x / (double)width * TAU;

            // Equation from http://graphicscodex.com  [sphry]
            d[0] =  sin(theta) * sin(phi);
            d[1] =               cos(theta);
            d[2] =  sin(theta) * cos(phi);
            d.normalize();

            vectToTexelCoordPanorama(d, width,  height, aU, aV ) ;
            const double solidAngle = texelPixelSolidAnglePanorama(aU, aV, width,  height);

            // Then compute the solid Angle of that thing
            //const double solidAngle = texelPixelSolidAngle(x, y, width,  height);
            ixy *= solidAngle;
            //r *= solidAngle;
            //g *= solidAngle;
            //b *= solidAngle;
            
#endif
            
            _sat[i] = ixy;

            _r[i] = r;
            _g[i] = g;
            _b[i] = b;

            //weightAccum += weight;
            //weightAccum += 1.0;
            weightAccum += solidAngle;
            _sum += ixy;
            
        }
    }
    // store for later use.
    _weightAccum = weightAccum;
    
    bool normalize = true;

    
    if (normalize){

        // normalize in order our image Accumulation exactly match 4 PI.
        const double normalizer = (4.0 * PI) / weightAccum;

        _sum *= normalizer;
        

        for (uint i = 0; i < imgSize; ++i) {

            _sat[i] *= normalizer;
            
            _minPonderedLum = std::min(_sat[i], _minPonderedLum);
            _maxPonderedLum = std::max(_sat[i], _maxPonderedLum);
            
        }
    }

#define ENHANCE_PRECISION 1
#ifdef ENHANCE_PRECISION

        // enhances precision of SAT
        // make values be around [0.0, 0.5]
        // https://developer.amd.com/wordpress/media/2012/10/SATsketch-siggraph05.pdf
        const double rangeLum = _maxLum - _minLum;
        const double rangePonderedLum = _maxPonderedLum -_minPonderedLum;        
        const double rangeR = _maxR - _minR;
        const double rangeG = _maxG - _minG;
        const double rangeB = _maxB - _minB;
        
        for (uint i = 0; i< imgSize; ++i) {
                
            _sat[i] = ((_sat[i] - _minPonderedLum) / rangePonderedLum) * 0.5;
            
            _r[i]  = ((_r[i] - _minR) / rangeR) * 0.5;
            _g[i]  = ((_g[i] - _minG) / rangeG) * 0.5;
            _b[i]  = ((_b[i] - _minB) / rangeB) * 0.5;
            
        }
#endif
    
        // now we sum
        for (uint y = 0; y < height; ++y) {
            for (uint x = 0; x < width;  ++x) {
                const uint i = y*width + x;
            
                // https://en.wikipedia.org/wiki/Summed_area_table
                _sat[i] = _sat[i] + I(x-1, y) + I(x, y-1) - I(x-1, y-1);

                _r[i]   = _r[i]   + R(x-1, y) + R(x, y-1) - R(x-1, y-1);
                _g[i]   = _g[i]   + G(x-1, y) + G(x, y-1) - G(x-1, y-1);
                _b[i]   = _b[i]   + B(x-1, y) + B(x, y-1) - B(x-1, y-1);

            }
        }


        // integral log
        for (uint y = 0; y < _height; ++y)
        {
            for (uint x = 0; x < _width;  ++x)
            {
                const uint i = y*width + x;

                double sum = I(x, y);
                if (sum > 0) sum = log(I(x, y));

                _sat1[i] = sum + I1(x-1, y) + I1(x, y-1) - I1(x-1, y-1);
            }
        }

        // Integral image of higher power
        // http://vision.okstate.edu/pubs/ssiai_tp_1.pdf
        // integral 2
        for (uint y = 0; y < _height; ++y)
        {
            for (uint x = 0; x < _width;  ++x)
            {
                const uint i = y*width + x;
                _sat2[i] = I(x, y)*I(x, y) + I2(x-1, y) + I2(x, y-1) - I2(x-1, y-1);
            }
        }
        // integral 3
        for (uint y = 0; y < _height; ++y) {
            for (uint x = 0; x < _width;  ++x) {
                const uint i = y*width + x;
                _sat3[i] = I(x, y)*I(x, y)*I(x, y) + I3(x-1, y) + I3(x, y-1) - I3(x-1, y-1);
            }
        }
        // integral 4
        for (uint y = 0; y < _height; ++y) {
            for (uint x = 0; x < _width;  ++x) {
                const uint i = y*width + x;
                _sat4[i] = I(x, y)*I(x, y)*I(x, y)*I(x, y) + I4(x-1, y) + I4(x, y-1) - I4(x-1, y-1);
            }
        }
        // integral 5
        for (uint y = 0; y < _height; ++y) {
            for (uint x = 0; x < _width;  ++x) {
                const uint i = y*width + x;
                _sat5[i] = I(x, y)*I(x, y)*I(x, y)*I(x, y) + I5(x-1, y) + I5(x, y-1) - I5(x-1, y-1);
            }
        }

}
