# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

add_mlir_dialect_library(MLIREsi
    EsiDialect.cpp
    EsiOps.cpp
    EsiTypes.cpp
    ${EsiDialactHeaders}

    DEPENDS
    MLIREsiDialectIncGen
    # MLIREsiIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIREDSC
    MLIRIR

    )
