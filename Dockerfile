FROM mcr.microsoft.com/dotnet/core/sdk:3.1-bionic
LABEL maintainer="John Demme (me@teqdruid.com)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install -y \
    man \
    g++ \
    clang \
    make \
    verilator
RUN apt-get install capnproto libcapnp-dev -y

CMD /bin/bash
