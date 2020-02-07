FROM mcr.microsoft.com/dotnet/core/sdk:3.1-bionic
LABEL maintainer="John Demme (me@teqdruid.com)"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install man -y
RUN apt-get install verilator -y
RUN apt-get install capnproto -y
RUN apt-get install g++ -y
RUN apt-get install clang -y
RUN apt-get install iverilog -y

CMD /bin/bash
