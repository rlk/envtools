
CXX=$(which g++-4.9)
CC=$(which gcc-4.9)
if [[ -z "${CXX}" ]]
then
    CXX=g++
    CC=gcc
fi


cmake ../ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC}
#cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_C_COMPILER=${CC}
