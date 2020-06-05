# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_mlir_dialect(EsiOps esi)
add_mlir_doc(EsiOps -gen-op-doc esi Dialects/)

set(EsiDialectHeaders EsiOps.hpp EsiOpsTypes.hpp)
