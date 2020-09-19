// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiTypes.hpp"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/TableGen/TypeDefGenHelpers.h>

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "Dialects/Esi/EsiDialectTypes.cpp.inc"
