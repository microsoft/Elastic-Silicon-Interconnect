
set(SOURCES ${CAPNP_SRCS} dummy.cpp)
add_executable(CosimDummySimulator ${HEADERS} ${SOURCES})
target_link_libraries(CosimDummySimulator
    EsiCosimDpiServer)
