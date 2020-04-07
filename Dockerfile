FROM mcr.microsoft.com/dotnet/core/sdk:3.1.201-bionic
LABEL maintainer="John Demme (me@teqdruid.com)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install apt-utils -y
RUN apt-get update && apt-get install -y \
    man \
    build-essential \
    ca-certificates \
    g++ \
    clang \
    make \
    python3 \
    python3-pip \
    git \
    capnproto libcapnp-dev \
    autoconf bc bison flex libfl-dev perl

RUN python3 -m pip install -U pylint
RUN python3 -m pip install -U pytest

# Compile Verilator so that we don't get a 3+ year old version
WORKDIR /tmp_verilator
ARG VERILATOR_REPO=https://github.com/verilator/verilator
ARG VERILATOR_SOURCE_COMMIT=v4.032
RUN git clone "${VERILATOR_REPO}" verilator && \
    cd verilator && \
    git checkout "${VERILATOR_SOURCE_COMMIT}" && \
    autoconf && \
    ./configure && \
    make -j "$(nproc)" && \
    make install && \
    cd .. && \
    rm -r verilator

WORKDIR /esi

CMD /bin/bash
