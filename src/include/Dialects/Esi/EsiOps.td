// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef ESI_OPS
#define ESI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "EsiDialect.td"

class Esi_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Esi_Dialect, mnemonic, traits>;


def Esi_CastCompound: Esi_Op<"cast_compound", [NoSideEffect]> {

}

#endif
