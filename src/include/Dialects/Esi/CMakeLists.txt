# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_mlir_dialect(EsiDialect esi)
add_mlir_doc(EsiDialect -gen-op-doc esi Dialects/)

set(EsiDialectHeaders EsiDialect.hpp EsiOps.hpp EsiTypes.hpp)
