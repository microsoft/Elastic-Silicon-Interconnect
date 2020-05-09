cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_BUILD_TYPE Debug)

# Pick up vcpkg toolchain from environment variable
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  message( STATUS "Found VCPKG at $ENV{VCPKG_ROOT}")
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

project(ElasticSiliconInterfaces VERSION 0.1.0)

if ( ${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR} )
    message( FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt." )
endif()

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set up CapnProto
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
find_package(CapnProto CONFIG REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src/core/cosim)
# add_subdirectory(src/core/cosim/dummy_simulator)

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
