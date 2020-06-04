
include_directories(${CAPNPC_OUTPUT_DIR})
include_directories(${CAPNP_INCLUDE_DIRS})
add_definitions(${CAPNP_DEFINITIONS})

capnp_generate_cpp(CAPNP_SRCS CANPN_HDRS esi_cosim_dpi.capnp)

include_directories(include)

set(HEADERS ${CAPNP_HDRS} include/svdpi.h include/dpi.hpp)
set(SOURCES ${CAPNP_SRCS} DpiEntryPoints.cpp Server.cpp EndPoint.cpp)
add_library(EsiCosimDpiServer SHARED ${HEADERS} ${SOURCES})
target_link_libraries(EsiCosimDpiServer PRIVATE
    CapnProto::kj CapnProto::kj-async CapnProto::kj-gzip
    CapnProto::capnp CapnProto::capnpc CapnProto::capnp-rpc 
    MtiPli )
    
install(TARGETS EsiCosimDpiServer
        LIBRARY DESTINATION /usr/lib )
