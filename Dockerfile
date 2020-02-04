FROM mcr.microsoft.com/dotnet/core/sdk:3.1-bionic
MAINTAINER John Demme (me@teqdruid.com)

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install verilator -y
RUN apt-get install capnproto -y

CMD /bin/bash