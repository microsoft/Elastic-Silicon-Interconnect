FROM mcr.microsoft.com/dotnet/core/sdk:3.1.201-bionic
LABEL maintainer="John Demme (me@teqdruid.com)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install apt-utils -y
RUN apt-get update && apt-get install -y \
    man \
    g++ \
    clang \
    make \
    verilator \
    python3 \
    python3-pip
RUN python3 -m pip install -U pylint
RUN python3 -m pip install -U pytest

RUN apt-get update && apt-get install capnproto libcapnp-dev -y

CMD /bin/bash
