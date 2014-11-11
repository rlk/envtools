
# to use openmp on osx, but you need to build gcc with brew
#CC=gcc-4.9
#CPP=g++-4.9
#CFLAGS= -std=c99 -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp
#CPPFLAGS= -mfpmath=sse -pedantic -Wall -O3 -m64 -fstrict-aliasing -fopenmp
#g++-4.9 -mfpmath=sse -fopenmp -O2 envtospecular.cpp -I/usr/local/include -L/usr/local/lib -ltiff -ljpeg -lpng -lz -lm

cmake_minimum_required(VERSION 2.8)
project(envtools)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/modules)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost COMPONENTS system )

find_package(PNG)
find_package(TIFF)
find_package(JPEG)

find_package(OpenImageIO)

include_directories(${OIIO_INCLUDE_DIR})
include_directories( ${TIFF_INCLUDE_DIR} )

add_executable(envremap envremap.c)
target_link_libraries(envremap ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} )

add_executable(envtoirr envtoirr.cpp Cubemap.cpp )
target_link_libraries(envtoirr ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} ${OIIO_LIBRARY} ${Boost_LIBRARIES} )

add_executable(fixedge fixEdge.cpp  Cubemap.cpp)
target_link_libraries(fixedge ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} ${OIIO_LIBRARY} ${Boost_LIBRARIES} )

add_executable(packer packer.cpp)
target_link_libraries(packer ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} ${OIIO_LIBRARY} ${Boost_LIBRARIES} )


# use the prefilter technic from UE4 but I think
# but something wrong with the rendering, so I keep it if I want to
# debug or update it later
add_executable(envtospecular envtospecular.cpp)
target_link_libraries(envtospecular ${PNG_LIBRARY} ${TIFF_LIBRARY} ${JPEG_LIBRARY} )
