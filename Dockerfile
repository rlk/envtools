FROM ubuntu:14.04.1

MAINTAINER trigrou

ENV LD_LIBRARY_PATH /usr/local/lib/:/usr/local/lib64/
ENV PYTHONPATH /usr/local/bin/:/usr/local/lib/python/site-packages

RUN echo "" >> /etc/apt/sources.list ;\
    echo "deb http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty universe" >>/etc/apt/sources.list ;\
    echo "deb-src http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty universe" >>/etc/apt/sources.list ;\
    echo "deb http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty multiverse" >>/etc/apt/sources.list ;\
    echo "deb-src http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty multiverse" >>/etc/apt/sources.list

RUN apt-get -y update --fix-missing && apt-get install -y \
    p7zip-full \
    ccache \
    cmake \
    g++ \
    git \
    libgif-dev \
    libwebp-dev \
    libpng12-dev \
    libtiff5-dev \
    libjpeg-dev \
    libopenjpeg-dev \
    libboost-dev libboost-filesystem-dev libboost-regex-dev libboost-system-dev libboost-thread-dev libboost-python-dev\
    software-properties-common \
    python \
    wget \
    libtbb-dev \
    python-pyopencl

RUN echo "/usr/local/lib64/" >/etc/ld.so.conf.d/lib64.conf
RUN echo "/usr/local/lib/" >/etc/ld.so.conf.d/lib.conf

RUN wget http://download.savannah.nongnu.org/releases/openexr/ilmbase-2.2.0.tar.gz
RUN tar xvfz ilmbase-2.2.0.tar.gz && cd ilmbase-2.2.0 && ./configure && make install

RUN wget http://download.savannah.nongnu.org/releases/openexr/openexr-2.2.0.tar.gz
RUN tar xvfz openexr-2.2.0.tar.gz && cd openexr-2.2.0 && ./configure --disable-ilmbasetest && make install

# openimageio
RUN cd /root/ && wget "https://github.com/OpenImageIO/oiio/archive/Release-1.5.16.tar.gz" \
&& tar xvfz Release-1.5.16.tar.gz && cd oiio-Release-1.5.16 && mkdir release && cd release/ \
&& cmake ../ -DCMAKE_BUILD_TYPE=Release && make install \
&& cd ../.. && rm -fr oiio-Release-1.5.16 && rm -fr Release-1.5.16.tar.gz

# envtools
#RUN rm -Rf /root/envtools
RUN mkdir /root/envtools
COPY ./ /root/envtools/

#RUN cd /root/ && git clone https://github.com/cedricpinson/envtools envtools && mkdir envtools/release && cd /root/envtools/release && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="ccache" -DCMAKE_CXX_COMPILER_ARG1="g++" ../
#RUN mkdir /root/envtools/release && cd /root/envtools/release && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="/usr/bin/clang++" ../ && make -j6 install

RUN mkdir /root/envtools/release && cd /root/envtools/release && cmake -DCMAKE_BUILD_TYPE=Release  ../ && make -j6 install

# RUN rm -r /root/envtools ilmbase-2.2.0.tar.gz ilmbase-2.2.0 openexr-2.2.0.tar.gz openexr-2.2.0

# OpenCL on HOST
# apt-get install build-essential
# apt-get install linux-image-extra-virtual
# wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
# mkdir nvidia_installers
# chmod +x cuda_7.5.18_linux.run
# ./cuda_7.5.18_linux.run -extract=`pwd`/nvidia_installers
# cd nvidia_installers/
# vi /etc/modprobe.d/blacklist-nouveau.conf
# echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
# update-initramfs -u
# apt-get install linux-headers-3.13.0-74-generic
# ./NVIDIA-Linux-x86_64-352.39.run



# OPENCL Docker
# needs to enable non official package
# RUN apt-get update && apt-get install -q -y \
# build-essential
# RUN echo "" >> /etc/apt/sources.list ;\
#     echo "deb http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty universe" >>/etc/apt/sources.list ;\
#     echo "deb-src http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty universe" >>/etc/apt/sources.list ;\
#     echo "deb http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty multiverse" >>/etc/apt/sources.list ;\
#     echo "deb-src http://us-west-2.ec2.archive.ubuntu.com/ubuntu/ trusty multiverse" >>/etc/apt/sources.list


# ENV CUDA_RUN http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run

# RUN cd /opt && \
#   wget $CUDA_RUN && \
#   chmod +x *.run && \
#   mkdir nvidia_installers && \
#   ./cuda_7.5.18_linux.run -extract=`pwd`/nvidia_installers && \
#   cd nvidia_installers && \
#   ./NVIDIA-Linux-x86_64-352.39.run -s -N --no-kernel-module && \
#   echo /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.352.39 >/etc/OpenCL/vendors/nvidia.icd &&
#   rm -r /opt/*
