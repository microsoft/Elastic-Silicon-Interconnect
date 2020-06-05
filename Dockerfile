FROM ubuntu:20.04
LABEL maintainer="John Demme (me@teqdruid.com)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install apt-utils -y
RUN apt-get update && apt-get install -y \
    man curl unzip tar \
    build-essential \
    ca-certificates \
    g++ gdb \
    clang lld \
    python3 python3-pip \
    python \
    git \
    autoconf bc bison flex libfl-dev perl \
    cmake make ninja-build

RUN python3 -m pip install -U pylint
RUN python3 -m pip install -U pytest
RUN python3 -m pip install -U cython
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install -U pycapnp


# Compile vcpkg to get cross-platform C/C++ library management
RUN cd / && \
    git clone https://github.com/Microsoft/vcpkg.git && \
    cd vcpkg && \
    git checkout d9b4acf02d8c784927f89d23edb677408ec428af
WORKDIR /vcpkg
RUN ./bootstrap-vcpkg.sh
ENV VCPKG_ROOT=/vcpkg

# Install libraries
RUN ./vcpkg install capnproto:x64-linux
ENV PATH=/vcpkg/installed/x64-linux/tools/capnproto/:$PATH

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

WORKDIR /llvm-project
ARG LLVM_REPO=https://github.com/llvm/llvm-project.git
# Arbitrary version -- change when needed
ARG LLVM_SOURCE_COMMIT=192cb718361dbd7be082bc0893f43bbc9782288f
RUN git clone "${LLVM_REPO}" . && \
    git checkout "${LLVM_SOURCE_COMMIT}"
RUN mkdir build
RUN cd build && cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
RUN cmake --build build --target check-mlir
ENV MLIR_DIR=/llvm-project/build/lib/cmake/mlir/

# Set up working environment
WORKDIR /esi
CMD /bin/bash
