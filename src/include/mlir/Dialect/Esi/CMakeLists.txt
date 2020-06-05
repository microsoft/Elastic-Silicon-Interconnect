# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_mlir_dialect(EsiOps esi)
add_mlir_doc(EsiOps -gen-op-doc esi Dialects/)

# set(LLVM_TARGET_DEFINITIONS EsiOps.td)
# mlir_tablegen(EsiOps.h.inc )
# mlir_tablegen(EsiOps.cpp.inc )
# add_public_tablegen_target(MLIREsiIncGen)
