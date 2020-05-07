FROM mcr.microsoft.com/dotnet/core/sdk:3.1.201-bionic
LABEL maintainer="John Demme (me@teqdruid.com)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install apt-utils -y
RUN apt-get update && apt-get install -y \
    man curl unzip tar \
    build-essential \
    ca-certificates \
    g++ gdb \
    clang \
    python3 \
    python3-pip \
    git \
    autoconf bc bison flex libfl-dev perl \
    cmake make

RUN python3 -m pip install -U pylint
RUN python3 -m pip install -U pytest
RUN python3 -m pip install -U cython
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install -U pycapnp


# Compile vcpkg to get cross-platform C/C++ library management
RUN cd / && \
    git clone https://github.com/Microsoft/vcpkg.git && \
    cd vcpkg && \
    git checkout 7db401cb1ef1fc559ec9f9ce814d064c328fd767
WORKDIR /vcpkg
RUN ./bootstrap-vcpkg.sh
ENV VCPKG_ROOT=/vcpkg

# Install libraries
RUN ./vcpkg install capnproto:x64-linux


# Compile Verilator so that we don't get a 3+ year old version
WORKDIR /verilator_src
ARG VERILATOR_REPO=https://github.com/verilator/verilator
ARG VERILATOR_SOURCE_COMMIT=v4.034
RUN git clone "${VERILATOR_REPO}" . && \
    git checkout "${VERILATOR_SOURCE_COMMIT}"
RUN autoconf && \
    ./configure && \
    make -j "$(nproc)" && \
    make install

# Set up working environment
WORKDIR /esi
CMD /bin/bash
