// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef ESI_DIALECT
#define ESI_DIALECT

include "mlir/IR/OpBase.td"

def Esi_Dialect : Dialect {
    let name = "esi";
    let cppNamespace = "esi";
}

#endif
