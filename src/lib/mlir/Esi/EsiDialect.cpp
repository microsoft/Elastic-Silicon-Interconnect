// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiOpsTypes.hpp"

namespace mlir {
namespace esi {

EsiDialect::EsiDialect (MLIRContext *context) :
    Dialect("esi", context) {
    addTypes<CompoundType>();
}

}
}
