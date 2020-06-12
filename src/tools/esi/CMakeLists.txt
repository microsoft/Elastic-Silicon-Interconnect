# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

  
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIREsi
        )
add_llvm_executable(esic esic.cpp)

llvm_update_compile_flags(esic)
target_link_libraries(esic PRIVATE ${LIBS})
