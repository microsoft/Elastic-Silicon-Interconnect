
add_definitions(${CAPNP_DEFINITIONS})

capnp_generate_cpp(CAPNP_SRCS CANPN_HDRS esi_cosim_dpi.capnp)

# set(HEADERS ${CAPNP_HDRS} ../include/svdpi.h ./include/dpi.hpp)
file(GLOB HEADERS ${COSIM_INCLUDE}/*.hpp)
set(SOURCES ${CAPNP_SRCS} DpiEntryPoints.cpp Server.cpp EndPoint.cpp)
add_library(EsiCosimDpiServer SHARED ${CAPNP_HDRS} ${HEADERS} ${SOURCES})
target_link_libraries(EsiCosimDpiServer PRIVATE
    CapnProto::kj CapnProto::kj-async CapnProto::kj-gzip
    CapnProto::capnp CapnProto::capnpc CapnProto::capnp-rpc 
    MtiPli )

target_include_directories(EsiCosimDpiServer PRIVATE ${CAPNPC_OUTPUT_DIR})
target_include_directories(EsiCosimDpiServer PRIVATE ${CAPNP_INCLUDE_DIRS})
target_include_directories(EsiCosimDpiServer PRIVATE ${COSIM_INCLUDE})
    
install(TARGETS EsiCosimDpiServer
        LIBRARY DESTINATION /usr/lib )
