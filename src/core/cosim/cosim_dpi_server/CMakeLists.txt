cmake_minimum_required(VERSION 3.0.0)
project(EsiCosimDpiServer VERSION 0.1.0)

# include(CTest)
# enable_testing()

include_directories(include)
set(HEADERS include/svdpi.h include/dpi_generated.h)
set(SOURCES EsiCosimDpiServer.cpp)
add_library(EsiCosimDpiServer SHARED ${HEADERS} ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
