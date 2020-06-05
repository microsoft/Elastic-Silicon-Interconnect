# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_mlir_dialect_library(EsiOps
    EsiOps.cpp

    DEPENDS
    MLIREsiOpsIncGen

    LINK_COMPONENTS
    Core
    )
