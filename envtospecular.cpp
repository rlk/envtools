/* Copyright (c) 2014 Cedric Pinson                                           */
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

#include <string>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdio>

#include <tiffio.h>

#include "gray.h"
#include "sRGB.h"

#define PI  3.1415927f


class Vec3f
{
    public:

        /** Data type of vector components.*/
        typedef float value_type;

        /** Number of vector components. */
        enum { num_components = 3 };

        value_type _v[3];

        /** Constructor that sets all components of the vector to zero */
        Vec3f() { _v[0]=0.0f; _v[1]=0.0f; _v[2]=0.0f;}
        Vec3f(value_type x,value_type y,value_type z) { _v[0]=x; _v[1]=y; _v[2]=z; }

        inline bool operator == (const Vec3f& v) const { return _v[0]==v._v[0] && _v[1]==v._v[1] && _v[2]==v._v[2]; }

        inline bool operator != (const Vec3f& v) const { return _v[0]!=v._v[0] || _v[1]!=v._v[1] || _v[2]!=v._v[2]; }

        inline bool operator <  (const Vec3f& v) const
        {
            if (_v[0]<v._v[0]) return true;
            else if (_v[0]>v._v[0]) return false;
            else if (_v[1]<v._v[1]) return true;
            else if (_v[1]>v._v[1]) return false;
            else return (_v[2]<v._v[2]);
        }

        inline value_type* ptr() { return _v; }
        inline const value_type* ptr() const { return _v; }

        inline void set( value_type x, value_type y, value_type z)
        {
            _v[0]=x; _v[1]=y; _v[2]=z;
        }

        inline void set( const Vec3f& rhs)
        {
            _v[0]=rhs._v[0]; _v[1]=rhs._v[1]; _v[2]=rhs._v[2];
        }

        inline value_type& operator [] (int i) { return _v[i]; }
        inline value_type operator [] (int i) const { return _v[i]; }

        inline value_type& x() { return _v[0]; }
        inline value_type& y() { return _v[1]; }
        inline value_type& z() { return _v[2]; }

        inline value_type x() const { return _v[0]; }
        inline value_type y() const { return _v[1]; }
        inline value_type z() const { return _v[2]; }

        /** Dot product. */
        inline value_type operator * (const Vec3f& rhs) const
        {
            return _v[0]*rhs._v[0]+_v[1]*rhs._v[1]+_v[2]*rhs._v[2];
        }

        /** Cross product. */
        inline const Vec3f operator ^ (const Vec3f& rhs) const
        {
            return Vec3f(_v[1]*rhs._v[2]-_v[2]*rhs._v[1],
                         _v[2]*rhs._v[0]-_v[0]*rhs._v[2] ,
                         _v[0]*rhs._v[1]-_v[1]*rhs._v[0]);
        }

        /** Multiply by scalar. */
        inline const Vec3f operator * (value_type rhs) const
        {
            return Vec3f(_v[0]*rhs, _v[1]*rhs, _v[2]*rhs);
        }

        /** Unary multiply by scalar. */
        inline Vec3f& operator *= (value_type rhs)
        {
            _v[0]*=rhs;
            _v[1]*=rhs;
            _v[2]*=rhs;
            return *this;
        }

        /** Divide by scalar. */
        inline const Vec3f operator / (value_type rhs) const
        {
            return Vec3f(_v[0]/rhs, _v[1]/rhs, _v[2]/rhs);
        }

        /** Unary divide by scalar. */
        inline Vec3f& operator /= (value_type rhs)
        {
            _v[0]/=rhs;
            _v[1]/=rhs;
            _v[2]/=rhs;
            return *this;
        }

        /** Binary vector add. */
        inline const Vec3f operator + (const Vec3f& rhs) const
        {
            return Vec3f(_v[0]+rhs._v[0], _v[1]+rhs._v[1], _v[2]+rhs._v[2]);
        }

        /** Unary vector add. Slightly more efficient because no temporary
          * intermediate object.
        */
        inline Vec3f& operator += (const Vec3f& rhs)
        {
            _v[0] += rhs._v[0];
            _v[1] += rhs._v[1];
            _v[2] += rhs._v[2];
            return *this;
        }

        /** Binary vector subtract. */
        inline const Vec3f operator - (const Vec3f& rhs) const
        {
            return Vec3f(_v[0]-rhs._v[0], _v[1]-rhs._v[1], _v[2]-rhs._v[2]);
        }

        /** Unary vector subtract. */
        inline Vec3f& operator -= (const Vec3f& rhs)
        {
            _v[0]-=rhs._v[0];
            _v[1]-=rhs._v[1];
            _v[2]-=rhs._v[2];
            return *this;
        }

        /** Negation operator. Returns the negative of the Vec3f. */
        inline const Vec3f operator - () const
        {
            return Vec3f (-_v[0], -_v[1], -_v[2]);
        }

        /** Length of the vector = sqrt( vec . vec ) */
        inline value_type length() const
        {
            return sqrtf( _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2] );
        }

        /** Length squared of the vector = vec . vec */
        inline value_type length2() const
        {
            return _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2];
        }

        /** Normalize the vector so that it has length unity.
          * Returns the previous length of the vector.
        */
        inline value_type normalize()
        {
            value_type norm = Vec3f::length();
            if (norm>0.0)
            {
                value_type inv = 1.0f/norm;
                _v[0] *= inv;
                _v[1] *= inv;
                _v[2] *= inv;
            }
            return( norm );
        }

};    // end of class Vec3f

class Vec3d
{
    public:

        /** Data type of vector components.*/
        typedef double value_type;

        /** Number of vector components. */
        enum { num_components = 3 };

        value_type _v[3];

        /** Constructor that sets all components of the vector to zero */
        Vec3d() { _v[0]=0.0; _v[1]=0.0; _v[2]=0.0;}

        inline operator Vec3f() const { return Vec3f(static_cast<float>(_v[0]),static_cast<float>(_v[1]),static_cast<float>(_v[2]));}

        Vec3d(value_type x,value_type y,value_type z) { _v[0]=x; _v[1]=y; _v[2]=z; }

        inline bool operator == (const Vec3d& v) const { return _v[0]==v._v[0] && _v[1]==v._v[1] && _v[2]==v._v[2]; }

        inline bool operator != (const Vec3d& v) const { return _v[0]!=v._v[0] || _v[1]!=v._v[1] || _v[2]!=v._v[2]; }

        inline bool operator <  (const Vec3d& v) const
        {
            if (_v[0]<v._v[0]) return true;
            else if (_v[0]>v._v[0]) return false;
            else if (_v[1]<v._v[1]) return true;
            else if (_v[1]>v._v[1]) return false;
            else return (_v[2]<v._v[2]);
        }

        inline value_type* ptr() { return _v; }
        inline const value_type* ptr() const { return _v; }

        inline void set( value_type x, value_type y, value_type z)
        {
            _v[0]=x; _v[1]=y; _v[2]=z;
        }

        inline void set( const Vec3d& rhs)
        {
            _v[0]=rhs._v[0]; _v[1]=rhs._v[1]; _v[2]=rhs._v[2];
        }

        inline value_type& operator [] (int i) { return _v[i]; }
        inline value_type operator [] (int i) const { return _v[i]; }

        inline value_type& x() { return _v[0]; }
        inline value_type& y() { return _v[1]; }
        inline value_type& z() { return _v[2]; }

        inline value_type x() const { return _v[0]; }
        inline value_type y() const { return _v[1]; }
        inline value_type z() const { return _v[2]; }


        /** Dot product. */
        inline value_type operator * (const Vec3d& rhs) const
        {
            return _v[0]*rhs._v[0]+_v[1]*rhs._v[1]+_v[2]*rhs._v[2];
        }

        /** Cross product. */
        inline const Vec3d operator ^ (const Vec3d& rhs) const
        {
            return Vec3d(_v[1]*rhs._v[2]-_v[2]*rhs._v[1],
                         _v[2]*rhs._v[0]-_v[0]*rhs._v[2] ,
                         _v[0]*rhs._v[1]-_v[1]*rhs._v[0]);
        }

        /** Multiply by scalar. */
        inline const Vec3d operator * (value_type rhs) const
        {
            return Vec3d(_v[0]*rhs, _v[1]*rhs, _v[2]*rhs);
        }

        /** Unary multiply by scalar. */
        inline Vec3d& operator *= (value_type rhs)
        {
            _v[0]*=rhs;
            _v[1]*=rhs;
            _v[2]*=rhs;
            return *this;
        }

        /** Divide by scalar. */
        inline const Vec3d operator / (value_type rhs) const
        {
            return Vec3d(_v[0]/rhs, _v[1]/rhs, _v[2]/rhs);
        }

        /** Unary divide by scalar. */
        inline Vec3d& operator /= (value_type rhs)
        {
            _v[0]/=rhs;
            _v[1]/=rhs;
            _v[2]/=rhs;
            return *this;
        }

        /** Binary vector add. */
        inline const Vec3d operator + (const Vec3d& rhs) const
        {
            return Vec3d(_v[0]+rhs._v[0], _v[1]+rhs._v[1], _v[2]+rhs._v[2]);
        }

        /** Unary vector add. Slightly more efficient because no temporary
          * intermediate object.
        */
        inline Vec3d& operator += (const Vec3d& rhs)
        {
            _v[0] += rhs._v[0];
            _v[1] += rhs._v[1];
            _v[2] += rhs._v[2];
            return *this;
        }

        /** Binary vector subtract. */
        inline const Vec3d operator - (const Vec3d& rhs) const
        {
            return Vec3d(_v[0]-rhs._v[0], _v[1]-rhs._v[1], _v[2]-rhs._v[2]);
        }

        /** Unary vector subtract. */
        inline Vec3d& operator -= (const Vec3d& rhs)
        {
            _v[0]-=rhs._v[0];
            _v[1]-=rhs._v[1];
            _v[2]-=rhs._v[2];
            return *this;
        }

        /** Negation operator. Returns the negative of the Vec3d. */
        inline const Vec3d operator - () const
        {
            return Vec3d (-_v[0], -_v[1], -_v[2]);
        }

        /** Length of the vector = sqrt( vec . vec ) */
        inline value_type length() const
        {
            return sqrt( _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2] );
        }

        /** Length squared of the vector = vec . vec */
        inline value_type length2() const
        {
            return _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2];
        }

        /** Normalize the vector so that it has length unity.
          * Returns the previous length of the vector.
          * If the vector is zero length, it is left unchanged and zero is returned.
        */
        inline value_type normalize()
        {
            value_type norm = Vec3d::length();
            if (norm>0.0)
            {
                value_type inv = 1.0/norm;
                _v[0] *= inv;
                _v[1] *= inv;
                _v[2] *= inv;
            }
            return( norm );
        }

};    // end of class Vec3d


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


static Vec3d cubemap_face[6][3] = {
    { Vec3d(0,0,-1), Vec3d(0,-1,0), Vec3d(1,0,0) },// x positif
    { Vec3d(0,0,1), Vec3d(0,-1,0), Vec3d(-1,0,0) }, // x negatif

    { Vec3d(1,0,0), Vec3d(0,0,1), Vec3d(0,1,0) },  // y positif
    { Vec3d(1,0,0), Vec3d(0,0,-1),Vec3d(0,-1,0) }, // y negatif

    { Vec3d(1,0,0), Vec3d(0,-1,0), Vec3d(0,0,1) },  // z positif
    { Vec3d(-1,0,0), Vec3d(0,-1,0),Vec3d(0,0,-1) } // z negatif
};

class Image
{
public:
    float* _images[6];
    std::string _name;
    uint16 _samplePerPixel;
    uint16 _bitsPerPixel;
    uint32 _width, _height;

    void loadEnvFace(TIFF* tif, int face);
    bool loadCubemap(const std::string& name) {

        TIFF *tif;
        tif = TIFFOpen(name.c_str(), "r");
        if (!tif)
            return false;


        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,  &_bitsPerPixel);
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,     &_width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH,    &_height);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &_samplePerPixel);

        for ( int face = 0; face < 6; ++face)
            loadEnvFace(tif, face);

        TIFFClose(tif);

        return true;
    }

    void iterateOnFace( int face ) {
        double xInvFactor = 2.0/double(_width);
        for ( int j = 0; j < _height; j++ ) {
            int lineIndex = j*_samplePerPixel*_width;
            for ( int i = 0; i < _width; i++ ) {
                int index = lineIndex + i*_samplePerPixel;
                Vec3d color = Vec3d( _images[face][ index ], _images[face][ index + 1 ], _images[face][ index +2 ] );
                // generate a vector for each texel
                Vec3d vecX = cubemap_face[face][0] * (double(i)*xInvFactor - 1.0);
                Vec3d vecY = cubemap_face[face][1] * (double(j)*xInvFactor - 1.0);
                Vec3d vecZ = cubemap_face[face][2];
                Vec3d direction = Vec3d( vecX + vecY + vecZ );
                direction.normalize();

#if 1
                Vec3d colorCheck;
                sample( direction, colorCheck );
                Vec3d diff = (color - colorCheck);
                if ( fabs(diff[0]) > 1e-6 || fabs(diff[1]) > 1e-6 || fabs(diff[2]) > 1e-6 ) {
                    std::cout << "face " << face << " " << i << "x" << j << " color error " << diff[0] << " " << diff[1] << " " << diff[2] << std::endl;
                    std::cout << "direction " << direction[0] << " " << direction[1] << " " << direction[2]  << std::endl;
                    return;
                }
#endif

            }
        }
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

    void sample(const Vec3d& direction, Vec3d& color ) {


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

        double sc = cubemap_face[index][0] * direction;
        double tc = cubemap_face[index][1] * direction;
        double ppx = (sc * maInv + 1.0) * 0.5 * _width; // width == height
        double ppy = (tc * maInv + 1.0) * 0.5 * _width; // width == height

        int px = int( floor( ppx +0.5 ) ); // center pixel
        int py = int( floor( ppy +0.5 ) ); // center pixel

        //std::cout << " px " << px << " py " << py << std::endl;

        int indexPixel = ( py * _width + px ) * _samplePerPixel;
        float r = _images[ index ][ indexPixel ];
        float g = _images[ index ][ indexPixel + 1 ];
        float b = _images[ index ][ indexPixel + 2 ];
        color[0] = r;
        color[1] = g;
        color[2] = b;
        //std::cout << "face " << index << " color " << r << " " << g << " " << b << std::endl;
    }

};


void Image::loadEnvFace(TIFF* tif, int face)
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

    if (! (w == _width && h == _height && b == _bitsPerPixel && c == _samplePerPixel)) {
        std::cerr << "can't read face " << face << std::endl;
        return;
    }

    std::cout << "reading face " << face << " " << w << " x " << h << " x " << c << std::endl;

    _images[face] = new float[_width*_height*_samplePerPixel];

    /* Allocate a scanline buffer. */
    std::vector<float> s;
    s.resize(TIFFScanlineSize(tif));

    uint16 k;
    /* Iterate over all pixels of the current cube face. */

    for (int i = 0; i < _height; ++i) {

        if (TIFFReadScanline(tif, &s.front(), i, 0) == 1) {

            for (int j = 0; j < _width; ++j) {

                for (int k =0; k < _samplePerPixel; k++) {
                    float p = s[ j * c + k ];
                    _images[face][(i*_width + j)*_samplePerPixel + k ] = p;
                }
            }
        }
    }
}

/*----------------------------------------------------------------------------*/

static int usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " in.tif out.tif" << std::endl;
    return 1;
}

int main(int argc, char *argv[])
{

    Image image;

    if ( argc < 2 )
        return usage(argv[0]);

    std::string in = std::string( argv[1] );
    image.loadCubemap(in);

    //Vec3d color;
    //image.sample( Vec3d(1,0,0), color);

    image.iterateOnFace(0);

    return 0;
}
