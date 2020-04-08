cmake_minimum_required(VERSION 3.0.0)
project(EsiCosimDpiServer VERSION 0.1.0)

if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
    message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

# include(CTest)
# enable_testing()
# include(FindCapnProto.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
find_package(CapnProto REQUIRED)
include_directories(${CAPNPC_OUTPUT_DIR})
add_definitions(${CAPNP_DEFINITIONS})

capnp_generate_cpp(CAPNP_SRCS CANPN_HDRS esi_cosim_dpi.capnp)

include_directories(include)
set(HEADERS ${CAPNP_HDRS} include/svdpi.h include/dpi.hxx)
set(SOURCES ${CAPNP_SRCS} EsiCosimDpiServer.cpp)
add_library(EsiCosimDpiServer SHARED ${HEADERS} ${SOURCES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
