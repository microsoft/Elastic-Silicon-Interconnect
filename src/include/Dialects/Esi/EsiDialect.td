// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef ESI_DIALECT
#define ESI_DIALECT

include "mlir/IR/OpBase.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Esi_Dialect : Dialect {
    let name = "esi";
    let cppNamespace = "esi";
}

def Esi_FixedType: Type<CPred<"$_self.isa<FixedPointType>()">, "ESI fixed point type">;

#endif
