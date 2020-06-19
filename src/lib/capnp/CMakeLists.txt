# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


add_definitions(${CAPNP_DEFINITIONS})

set(HEADERS )# ${EsiCapnpHeaders})
set(SOURCES ConvertToESI.cpp)
add_mlir_library(CapnpEsi SHARED ${HEADERS} ${SOURCES})
target_link_libraries(CapnpEsi PUBLIC
    CapnProto::kj
    CapnProto::capnp
    # CapnProto::capnpc
    MLIREsi
    )
