
# to use openmp on osx, but you need to build gcc with brew
#CC=gcc-4.9
#CPP=g++-4.9
#CFLAGS= -std=c99 -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp
#CPPFLAGS= -mfpmath=sse -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp
#g++-4.9 -mfpmath=sse -fopenmp -O2 envtospecular.cpp -I/usr/local/include -L/usr/local/lib -ltiff -ljpeg -lpng -lz -lm

cmake_minimum_required(VERSION 2.8)
project(envtools)


find_package(PNG)
find_package(TIFF)
find_package(JPEG)

include_directories( ${TIFF_INCLUDE_DIR} )

add_executable(envremap envremap.c)
target_link_libraries(envremap ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} )

add_executable(envtoirr envtoirr.cpp Cubemap.cpp )
target_link_libraries(envtoirr ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} )

# use the prefilter technic from UE4 but I think
# but something wrong with the rendering, so I keep it if I want to
# debug or update it later
add_executable(envtospecular envtospecular.cpp)
target_link_libraries(envtospecular ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} )
