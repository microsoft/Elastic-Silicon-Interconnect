
include_directories(../cosim_dpi_server/include)

# include_directories(include)

set(HEADERS ${CAPNP_HDRS} ../cosim_dpi_server/include/svdpi.h)
set(SOURCES ${CAPNP_SRCS} DummySvDpi.cpp )
add_library(MtiPli SHARED ${HEADERS} ${SOURCES})


