// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiOpsTypes.hpp"
#include "Dialects/Esi/EsiOps.hpp"

namespace mlir {
namespace esi {

EsiDialect::EsiDialect (MLIRContext *context) :
    Dialect("esi", context) {
    addTypes<CompoundType>();

    addOperations<
    #define GET_OP_LIST
    #include "Dialects/Esi/EsiOps.cpp.inc"
    >();
}

}
}
