/* -*-c++-*- */

#include "SummedAreaTableRegion"
#include "SummedAreaTable"

/**
 * A subregion in a SummedAreaTable.
 */
double SatRegion::getRegionWeight() const
{

    double f00 = _sat->I1(_x,_y);
    double f10 = _sat->I1(_x+_w,_y);
    double f01 = _sat->I1(_x,_y+_h);
    double f11 = _sat->I1(_x+_w,_y+_h);

    return _sum * (f00 + f10 + f01 + f11) * 0.25;
}

double SatRegion::getCornerDeviation() const
{
    const Vec2d c = centroid();
    const double fav = _sat->I1(c[0],c[1]);


    double f00fav = _sat->I1(_x,_y) - fav;
    double f10fav = _sat->I1(_x+ _w,_y) - fav;
    double f01fav = _sat->I1(_x,_y+_h) - fav;
    double f11fav = _sat->I1(_x+ _w,_y+_h) - fav;

    f00fav *= f00fav;
    f10fav *= f10fav;
    f01fav *= f01fav;
    f11fav *= f11fav;

    return 0.5 * sqrt(f00fav + f10fav + f01fav + f11fav);
}

void SatRegion::create(const int x, const int y, const uint w, const uint h, const SummedAreaTable* sat)
{
    _x = x; _y = y; _w = w; _h = h; _sat = sat;

    _sum = _sat->sum(x,       y,
                     x+(w-1), y,
                     x+(w-1), y+(h-1),
                     x,       y+(h-1));

    _r = _sat->sumR(x,       y,
                    x+(w-1), y,
                    x+(w-1), y+(h-1),
                    x,       y+(h-1));

    _g = _sat->sumG(x,       y,
                    x+(w-1), y,
                    x+(w-1), y+(h-1),
                    x,       y+(h-1));

    _b = _sat->sumB(x,       y,
                    x+(w-1), y,
                    x+(w-1), y+(h-1),
                    x,       y+(h-1));


    _sum2 = _sat->sum2(x,       y,
                       x+(w-1), y,
                       x+(w-1), y+(h-1),
                       x,       y+(h-1));

    _sum3 = _sat->sum3(x,       y,
                       x+(w-1), y,
                       x+(w-1), y+(h-1),
                       x,       y+(h-1));

    _sum4 = _sat->sum4(x,       y,
                       x+(w-1), y,
                       x+(w-1), y+(h-1),
                       x,       y+(h-1));

    _sum5 = _sat->sum5(x,       y,
                       x+(w-1), y,
                       x+(w-1), y+(h-1),
                       x,       y+(h-1));



}
