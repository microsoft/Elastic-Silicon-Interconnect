# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_mlir_dialect_library(EsiOps
    EsiDialect.cpp
    EsiOps.cpp
    ${EsiDialactHeaders}

    DEPENDS
    MLIREsiOpsIncGen
    # MLIREsiIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIREDSC
    MLIRIR

    )
