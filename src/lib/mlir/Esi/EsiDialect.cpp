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
    addTypes<CompoundType>();

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
    if (typeKeyword == CompoundType::getKeyword())
        return CompoundType::parse(getContext(), parser);
}

/// Print a type registered to this dialect
void EsiDialect::printType(Type type, DialectAsmPrinter &printer) const {
    switch (type.getKind())
    {
        case Types::Compound: {
            auto c = type.dyn_cast<CompoundType>();
            c.print(printer);
            break;
        }
    }
}

}
}
