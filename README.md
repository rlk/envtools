# Environment Mapping Tools

Copyright &copy; 2012&ndash;2013 &mdash; [Robert Kooima](http://kooima.net)

The environment mapping tools are a set of command line utilities that operate upon spherical images in TIFF form. These operations including remapping between common spherical projections and generating diffuse irradiance maps from specular environment maps.

- [envremap.c](envremap.c)
- [envtoirr.c](envtoirr.c)

The following header files provide the ICC profiles needed for well-formed floating-point TIFFs.

- [gray.h](gray.h)
- [sRGB.h](sRGB.h)

## Spherical Remapping

This code supports reprojection and resampling between any two of the following spherical image projections. A high-resolution example image of each type is included.

<table>
    <tr><td><img src="etc/thumbnail-rect.png"></td><td>The <b>rect</b> type corresponds to the equirectangular projection, much like the common map of the world. It represents the full sphere, though with significant distortion near the poles. (<a href="etc/rect.tif">Example</a>.)</td></tr>

    <tr><td><img src="etc/thumbnail-ball.png"></td><td>The <b>ball</b> type corresponds to a photograph of a mirrored sphere, or "light probe". It represents the full sphere, but with significant distortion toward the back. (<a href="etc/ball.tif">Example</a>.)</td></tr>

    <tr><td><img src="etc/thumbnail-dome.png"></td><td>The <b>dome</b> type gives a "dome master". This is an image suitable for projection onto a fulldome planetarium. The view looks up and the radius varies linearly with latitude. It represents only half of the sphere. (<a href="etc/dome.tif">Example</a>.)</td></tr>

    <tr><td><img src="etc/thumbnail-hemi.png"></td><td>The <b>hemi</b> type is mathematically identical to the dome type, though the view faces forward instead of up. This corresponds to a photograph taken with an 8mm "fisheye" lens. This too represents only half of the sphere. (<a href="etc/hemi.tif">Example</a>.)</td></tr>

    <tr><td><img src="etc/thumbnail-cube.png"></td><td>The <b>cube</b> type corresponds to an OpenGL cube map texture, and is the best choice for use in real-time 3D rendering. The TIFF contains each of the six cube faces in a separate page. The cube faithfully represents the full sphere with minimal distortion. (<a href="etc/cube.tif">Example</a>.)</td></tr>
</table>

The output is sampled using one of several sampling patterns, which give a quality-speed tradeoff.

<table>
    <tr><td><img src="etc/cent.png"></td><td><b>cent</b> &hellip; One sample at the centroid of the output pixel</td></tr>
    <tr><td><img src="etc/rgss.png"></td><td><b>rgss</b> &hellip; Rotated-grid super sampling</td></tr>
    <tr><td><img src="etc/box2.png"></td><td><b>box2</b> &hellip; 2 &times;2 super sampling</td></tr>
    <tr><td><img src="etc/box3.png"></td><td><b>box3</b> &hellip; 3 &times;3 super sampling</td></tr>
    <tr><td><img src="etc/box4.png"></td><td><b>box4</b> &hellip; 4 &times;4 super sampling</td></tr>
</table>

This tool remaps the input image `src.tif` to the output `dst.tif`. The sample depth and format of the input TIFF is preserved in the output.

`envremap [-i input] [-o output] [-p pattern] [-f filter] [-n n] src.tif dst.tif`

- `-i input`

    Input projection type. May be `ball`, `cube`, `dome`, `hemi`, or `rect`. The default is `rect`.

- `-o output`

    Output projection type. May be `ball`, `cube`, `dome`, `hemi`, or `rect`. The default is `rect`.

- `-p pattern`

    Output sampling pattern. May be `cent`, `rgss`, `box2`, `box3`, or `box4`. The default is `rgss`.

- `-f filter`

    Input filter type. Maybe `nearest` or `linear`. The default is `linear`.

- `-n n`

    Output size. Image will have size `n` &times; `n`, except `rect` which will have size 2`n` &times; `n`.

## Irradiance Generation

This tool generates an irradiance environment map from a given environment map using the method of Ramamoorthi and Hanrahan's [An Efficient Representation for Irradiance Environment Maps](http://graphics.stanford.edu/papers/envmap/). The input _must_ be a 32-bit floating point TIFF image in cube map format with six pages.

`envtoirr [-n n] [-f src.tif] [-ges] dst.tif`

- `-n n`

    Output size. The output will be a 32-bit floating point TIFF with six pages, each `n` &times; `n` in size.

- `-f src.tif`

    Input TIFF file.

- `-g`  
  `-e`  
  `-s`

    Forego loading an input file and instead generate an irradiance map using one of the parameter sets from Ramamoorth and Hanrahan. Options are `-g` Grace Cathedral, `-e` Eucalyptus Grove, or `s` St. Peter's Basilica.
