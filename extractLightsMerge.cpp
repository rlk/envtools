/* -*-c++-*- */

/**
 * convert Env map Regions to Lights
 */
void createLightsFromRegions(const SatRegionVector& regions, LightVector& lights, const float *rgba, const int width, const int height, const int nc)
{
    // convert region into lights
    for (SatRegionVector::const_iterator region = regions.begin(); region != regions.end(); ++region)
    {

        Light l;

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

        const uint i = static_cast<uint>(l._centroidPosition[1]*width + l._centroidPosition[0]);

        double r = rgba[i*nc + 0];
        double g = rgba[i*nc + 1];
        double b = rgba[i*nc + 2];
        l._luminancePixel = luminance(r,g,b);

        lights.push_back(l);
    }

}

/*
 *  Merge a light into another and store a copy inside the parent
 */
void mergeLight(Light& lightParent, Light& lightChild)
{
    // exclude from next merges
    lightChild._merged = true;

    const int x = lightParent._x;
    const int y = lightParent._y;
    const int w = lightParent._w;
    const int h = lightParent._h;

    lightParent._x = std::min(x, (int)lightChild._x);
    lightParent._y = std::min(y, (int)lightChild._y);

    lightParent._w = std::max(x + w, (int)(lightChild._x + lightChild._w)) - lightParent._x;
    lightParent._h = std::max(y + h, (int)(lightChild._y + lightChild._h)) - lightParent._y;


    lightParent.childrenLights.push_back(lightChild);

    //lightChild._sortCriteria = lightChild._lumAverage;
    //lightChild._sortCriteria = lightChild._sum;
    lightChild._sortCriteria = lightChild._areaSize;

    // light is bigger, better candidate to main light
    lightParent._mergedNum++;

    lightParent._sum += lightChild._sum;

    double newAreaSize = lightParent._areaSize+lightChild._areaSize;

    double ratioParent = lightParent._areaSize / newAreaSize;
    double ratioChild = lightChild._areaSize / newAreaSize;

    lightParent._rAverage = lightParent._rAverage*ratioParent + lightChild._rAverage*ratioChild;
    lightParent._gAverage = lightParent._gAverage*ratioParent + lightChild._gAverage*ratioChild;
    lightParent._bAverage = lightParent._bAverage*ratioParent + lightChild._bAverage*ratioChild;

    lightParent._areaSize = newAreaSize;
    lightParent._lumAverage = lightParent._sum / newAreaSize;

}

/*
 * not a constructor, not struct member as it's specific for merge
 *  we copy only parts
 */
void lightCopy(Light &lDest, const Light &lSrc)
{
    lDest._merged = false;
    lDest._mergedNum = 0;
    lDest._x = lSrc._x;
    lDest._y = lSrc._y;
    lDest._w = lSrc._w;
    lDest._h = lSrc._h;
    lDest._centroidPosition = lSrc._centroidPosition;
    lDest._areaSize = lSrc._areaSize;
    lDest._sum = lSrc._sum;
    lDest._variance = lSrc._variance;
    lDest._lumAverage = lSrc._lumAverage;
    lDest._rAverage = lSrc._rAverage;
    lDest._gAverage = lSrc._gAverage;
    lDest._bAverage = lSrc._bAverage;
    lDest._luminancePixel = lSrc._luminancePixel;
}
/*
 * intersect lights with another light
 */
bool intersectLightAgainstLights2D(const LightVector &lights, const Light &lightCandidate, int border)
{
    if (lights.empty()) return true;


    int x1 = lightCandidate._x - border;
    int y1 = lightCandidate._y - border;
    int x2 = lightCandidate._x + lightCandidate._w + border;
    int y2 = lightCandidate._y + lightCandidate._h + border;

    for (LightVector::const_iterator l = lights.begin(); l != lights.end(); ++l)
    {
        if (!(l->_y > y2 || l->_y+l->_h < y1 || l->_x > x2 || l->_x+l->_w < x1))
        {
            return true;
        }
    }

    return false;
}

/**
 * Merge small area light neighbour with small area light neighbours
 */
uint mergeLights(LightVector& lights, LightVector& newLights, const uint width, const uint height, const double areaSizeMax, const double luminanceMaxLight)
{

    // discard or keep Light too near an current light
    const int border = 5;//static_cast <uint> (sqrt (areaSizeMax) / 2.0);

    // for each light we try to merge with all other intersecting lights
    // that are in the same neighbourhood of the sorted list of lights
    // where neighbour are of near same values
    for (LightVector::iterator lightIt = lights.begin(); lightIt != lights.end(); ++lightIt)
    {

        // already merged in a previous light
        // we do nothing
        if (lightIt->_merged) continue;

        // ignore too big, which is the same as ignoring low lum lights
        if (areaSizeMax < lightIt->_areaSize) break;

        Light lCurrent;
        lightCopy(lCurrent, *lightIt);
        int x1 = lCurrent._x - border;
        int y1 = lCurrent._y - border;
        int x2 = lCurrent._x + lCurrent._w + border;
        int y2 = lCurrent._y + lCurrent._h + border;

        uint numMergedLight;

        // current area Size will change when getting merges
        // we store initial values to prevent merging
        // with light too low
        const double areaSizeCurrent = lCurrent._areaSize;

        do {

            numMergedLight = 0;

            // could start at current light
            for (LightVector::iterator l = lights.begin(); l != lights.end(); ++l) {

                // ignore already merged or itself
                if (l->_merged || l == lightIt ) continue;

                // ignore too big, which is the same as ignoring low lum lights
                // (subsequent in vector are all bigger, so stop here)
                //if (areaSizeMax < lCurrent._areaSize + l->_areaSize) break;
                if (areaSizeMax < l->_areaSize) break;

                bool intersect2D = !(l->_y > y2 || l->_y+l->_h < y1 || l->_x > x2 || l->_x+l->_w < x1);
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
                if (intersect2D)
                {

                    mergeLight(lCurrent, *l);

                    x1 = lCurrent._x - border;
                    y1 = lCurrent._y - border;
                    x2 = lCurrent._x + lCurrent._w + border;
                    y2 = lCurrent._y + lCurrent._h + border;

                    numMergedLight++;
                }

            }

            // if we're merging we're changing borders
            // means we have new neighbours
            // or light now included inside our area
        } while (numMergedLight > 0);

        if (lCurrent._mergedNum > 0){

            //lCurrent._sortCriteria = lCurrent._lumAverage;
            lCurrent._sortCriteria = lCurrent._areaSize;

            newLights.push_back(lCurrent);

            // add new light to current one too
            // to allow for merged light merge ?
            //lights.push_back(lCurrent);

        }

    }

    // count merged light
    uint numMergedLightTotal = 0;
    for (LightVector::iterator lCurrent = newLights.begin(); lCurrent != newLights.end(); ++lCurrent)
    {
        if (!lCurrent->_merged) numMergedLightTotal += lCurrent->_mergedNum;
    }

    // fill new array with light that wasn't merged at all
    for (LightVector::iterator lCurrent = lights.begin(); lCurrent != lights.end(); ++lCurrent)
    {
        // add remaining non merged lights
        if (!lCurrent->_merged && lCurrent->_mergedNum == 0){

            lCurrent->_lumAverage = lCurrent->_sum / lCurrent->_areaSize;

            //lCurrent->_sortCriteria = lCurrent->_lumAverage;
            lCurrent->_sortCriteria = lCurrent->_sum;

            newLights.push_back(*lCurrent);
        }
    }

    return numMergedLightTotal;

}


// we now have merged big area light, which are the zone with the most light possible
// now we mush restrict those to smallest possible significant light possible
// reducing it to a near directional light as much as possible
uint selectLights(LightVector& mergedLights, LightVector& newLights, const uint width, const uint height, const double areaSizeMax, const double luminanceMaxLight, const  double envLuminanceSum)
{

    // discard or keep light too near an current light
    const uint border = 1;//static_cast <uint> (sqrt (areaSizeMax) / 2.0);
    uint numMergedLightTotal = 0;


    for (LightVector::iterator lightIt = mergedLights.begin(); lightIt != mergedLights.end(); ++lightIt)
    {

        // already merged, we find in lights the light intersecting
        uint numMergedLight = 0;
        // if light "splittable"
        // and light already over ratio, need to cut it
        if (lightIt->_mergedNum > 0)
        {

            LightVector &lights = lightIt->childrenLights;

            // sort to get most powerful light first
            std::sort(lights.begin(), lights.end());


            // take biggest and merge a bit
            Light lCurrent;
            lightCopy(lCurrent, lights[0]);
            int x1 = lCurrent._x - border;
            int y1 = lCurrent._y - border;
            int x2 = lCurrent._x + lCurrent._w + border;
            int y2 = lCurrent._y + lCurrent._h + border;

            // reset children lights to start over merge process
            for (LightVector::iterator l = lights.begin()+1; l != lights.end(); ++l)
            {
                l->_merged = false;
            }

            do {
                numMergedLight = 0;

                for (LightVector::iterator l = lights.begin()+1; l != lights.end(); ++l)
                {

                    // ignore already merged or itself
                    if (l->_merged ) continue;

                    // pick only smallest & near lights , grow until not bigger than areaSize
                    if (areaSizeMax <= l->_areaSize) {
                        break;
                    }

                    bool intersect2D = !(l->_y > y2 || l->_y+l->_h < y1 || l->_x > x2 || l->_x+l->_w < x1);

                    if (intersect2D && intersectLightAgainstLights2D(lCurrent.childrenLights, *l, border))
                    {

                        mergeLight(lCurrent, *l);

                        x1 = lCurrent._x - border;
                        y1 = lCurrent._y - border;
                        x2 = lCurrent._x + lCurrent._w + border;
                        y2 = lCurrent._y + lCurrent._h + border;

                        // if too big, stop
                        if (areaSizeMax <= lCurrent._areaSize) {
                            break;
                        }

                        // if too luminous, stop
                        // doesn't work
                        //if (luminanceMaxLight < lCurrent._sum) {
                        //    break;
                        //}

                        numMergedLight++;
                    }


                }
            } while (numMergedLight > 0);


            //lCurrent._sortCriteria = lCurrent._lumAverage;
            //lCurrent._sortCriteria = lCurrent._variance;
            //lCurrent._sortCriteria = lCurrent._areaSize;
            lCurrent._sortCriteria = lCurrent._sum;
            newLights.push_back(lCurrent);

            numMergedLightTotal += lCurrent._mergedNum;

        }
        else
        {

            lightIt->_sortCriteria = lightIt->_lumAverage;
            lightIt->_sortCriteria = lightIt->_sum;
            newLights.push_back(*lightIt);
            numMergedLightTotal += lightIt->_mergedNum;

        }
    }


    return numMergedLightTotal;

}
