// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Esi_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Esi_Dialect, mnemonic, traits>;


def Esi_CastCompound: Esi_Op<"cast_compound", [NoSideEffect]> {
    let arguments = (ins Esi_Type:$value);
    let results = (outs AnyI1:$result);
}
