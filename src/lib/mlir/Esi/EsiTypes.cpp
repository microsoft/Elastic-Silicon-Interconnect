// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiTypes.hpp"
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;

namespace mlir {
namespace esi {

Type CompoundType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    bool isSigned;
    unsigned whole;
    unsigned fractional;

    if (parser.parseLess()) return Type();
    StringRef isSignedStr;
    if (parser.parseKeyword(&isSignedStr)) return Type();
    if (isSignedStr.compare_lower("true")) isSigned = true;
    else if (isSignedStr.compare_lower("false")) isSigned = false;
    else {
        parser.emitError(parser.getCurrentLocation(), "Expected true or false");
        return nullptr;
    }

    if (parser.parseComma()) return Type();
    if (parser.parseInteger(whole)) return Type();
    if (parser.parseComma()) return Type();
    if (parser.parseInteger(fractional)) return Type();
    if (parser.parseGreater()) return Type();

    return get(ctxt, isSigned, whole, fractional);
}

void CompoundType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getKeyword() << "<"
        << ( getImpl()->isSigned ? "true" : "false" ) << ","
        << getImpl()->whole << ","
        << getImpl()->fractional << ">";
}

}
}
