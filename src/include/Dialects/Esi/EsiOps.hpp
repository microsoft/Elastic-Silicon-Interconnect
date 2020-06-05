// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_OPS_HPP__
#define __ESI_OPS_HPP__

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiOpsTypes.hpp"

#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir {
namespace esi {

#define GET_OP_CLASSES
#include "Dialects/Esi/EsiOps.h.inc"

}
}

#endif
