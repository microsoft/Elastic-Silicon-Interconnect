// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef ESI_OPS
#define ESI_OPS

include "mlir/IR/OpBase.td"
include "EsiDialect.td"

class Esi_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Esi_Dialect, mnemonic, traits>;

#endif
