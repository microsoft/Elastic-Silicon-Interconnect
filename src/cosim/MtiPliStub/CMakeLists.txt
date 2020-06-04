
set(HEADERS ${CAPNP_HDRS} ${COSIM_INCLUDE}/svdpi.h)
set(SOURCES ${CAPNP_SRCS} DummySvDpi.cpp )
add_library(MtiPli SHARED ${HEADERS} ${SOURCES})
target_include_directories(MtiPli PRIVATE ${COSIM_INCLUDE})

