
set(HEADERS ${CAPNP_HDRS} )
set(SOURCES ${CAPNP_SRCS} dummy.cpp)
add_executable(CosimDummySimulator ${HEADERS} ${SOURCES})
target_link_libraries(CosimDummySimulator
    EsiCosimDpiServer)

target_include_directories(CosimDumm COSIM_INCLUDE)