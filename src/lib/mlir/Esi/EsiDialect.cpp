// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiTypes.hpp"
#include "Dialects/Esi/EsiOps.hpp"

#include <mlir/IR/DialectImplementation.h>

namespace mlir {
namespace esi {

EsiDialect::EsiDialect (MLIRContext *context) :
    Dialect("esi", context) {
    addTypes<
        FixedPointType,
        FloatingPointType
      >();

    addOperations<
    #define GET_OP_LIST
    #include "Dialects/Esi/EsiOps.cpp.inc"
    >();
}


/// Parses a type registered to this dialect
Type EsiDialect::parseType(DialectAsmParser &parser) const {
    llvm::StringRef typeKeyword;
    if (parser.parseKeyword(&typeKeyword))
        return Type();
    if (typeKeyword == FixedPointType::getKeyword())
        return FixedPointType::parse(getContext(), parser);
    if (typeKeyword == FloatingPointType::getKeyword())
        return FloatingPointType::parse(getContext(), parser);
    return Type();
}

/// Print a type registered to this dialect
void EsiDialect::printType(Type type, DialectAsmPrinter &printer) const {
    switch (type.getKind())
    {
        case Types::FixedPoint: {
            auto c = type.dyn_cast<FixedPointType>();
            c.print(printer);
            break;
        }
        case Types::FloatingPoint: {
            auto c = type.dyn_cast<FloatingPointType>();
            c.print(printer);
            break;
        }
    }
}

}
}
