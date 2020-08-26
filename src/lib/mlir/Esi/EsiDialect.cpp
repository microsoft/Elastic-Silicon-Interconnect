// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiTypes.hpp"
#include "Dialects/Esi/EsiOps.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>

namespace mlir {
namespace esi {

EsiDialect::EsiDialect (MLIRContext *context) :
    Dialect("esi", context, TypeID::get<EsiDialect>()) {
    addTypes<
        // FixedPointType,
        // FloatingPointType,
        // ListType,
        // StructType,
        // UnionType,
        // EnumType,
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
    llvm::errs() << llvm::formatv("Could not parse esi.{0}!\n", mnemonic);
    return Type();
}

/// Print a type registered to this dialect
void EsiDialect::printType(Type type, DialectAsmPrinter &printer) const {
    if (!generatedTypePrinter(type, printer))
        return;
    TypeSwitch<Type>(type)
        .Default([](Type) { llvm_unreachable("unexpected 'esi' type kind"); });

    // switch (type.getKind())
    // {
    //     case Types::FixedPoint: {
    //         auto c = type.dyn_cast<FixedPointType>();
    //         c.print(printer);
    //         break;
    //     }
    //     case Types::FloatingPoint: {
    //         auto c = type.dyn_cast<FloatingPointType>();
    //         c.print(printer);
    //         break;
    //     }
    //     case Types::List: {
    //         auto c = type.dyn_cast<ListType>();
    //         c.print(printer);
    //         break;
    //     }
    //     case Types::Struct: {
    //         auto c = type.dyn_cast<StructType>();
    //         c.print(printer);
    //         break;
    //     }
    //     case Types::Union: {
    //         auto c = type.dyn_cast<UnionType>();
    //         c.print(printer);
    //         break;
    //     }
    //     case Types::Enum: {
    //         auto c = type.dyn_cast<EnumType>();
    //         c.print(printer);
    //         break;
    //     }
    // }
}

}
}
