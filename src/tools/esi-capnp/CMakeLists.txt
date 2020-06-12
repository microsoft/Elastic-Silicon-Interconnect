# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

  
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_llvm_executable(esi-capnp esi-capnp.cpp)
target_link_libraries(esi-capnp PRIVATE
    CapnpEsi
    MLIREsi
#     CapnProto::kj
#     CapnProto::capnp
    ${dialect_libs}
)
