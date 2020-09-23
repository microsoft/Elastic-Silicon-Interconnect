// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiOps.hpp"
#include "Dialects/Esi/EsiTypes.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/DialectImplementation.h>

namespace mlir {
namespace esi {

EsiDialect::EsiDialect(MLIRContext *context)
    : Dialect("esi", context, TypeID::get<EsiDialect>()) {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialects/Esi/EsiDialectTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialects/Esi/EsiDialect.cpp.inc"
      >();
}

/// Parses a type registered to this dialect
Type EsiDialect::parseType(DialectAsmParser &parser) const {
  llvm::StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();
  auto genType = generatedTypeParser(getContext(), parser, mnemonic);
  if (genType != Type())
    return genType;
  parser.emitError(parser.getCurrentLocation(),
                   llvm::formatv("Could not parse esi.{0}!\n", mnemonic));
  return Type();
}

/// Print a type registered to this dialect
void EsiDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (!generatedTypePrinter(type, printer))
    return;
  llvm_unreachable("unexpected 'esi' type kind");
}

} // namespace esi
} // namespace mlir
