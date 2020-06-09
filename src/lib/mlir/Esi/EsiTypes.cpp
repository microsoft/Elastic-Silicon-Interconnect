// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiTypes.hpp"
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;

namespace mlir {
namespace esi {
namespace details {

    struct FractionalTypeStorage : public TypeStorage {
        FractionalTypeStorage(bool isSigned, unsigned whole, unsigned fractional)
            : isSigned(isSigned), whole(whole), fractional(fractional) { }

        /// The hash key for this storage is a pair of the integer and type params.
        using KeyTy = std::tuple<bool, unsigned, unsigned>;

        /// Define the comparison function for the key type.
        bool operator==(const KeyTy &key) const {
            return key == KeyTy(isSigned, whole, fractional);
        }

        static llvm::hash_code hashKey(const KeyTy &key) {
            auto [isSigned, whole, fractional] = key;
            return llvm::hash_combine(isSigned, whole, fractional);
        }

        /// Define a construction method for creating a new instance of this storage.
        static FractionalTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
            auto [isSigned, whole, fractional] = key;
            return new (allocator.allocate<FractionalTypeStorage>())
                FractionalTypeStorage(isSigned, whole, fractional);
        }

        bool isSigned;
        unsigned whole;
        unsigned fractional;
    };

}

FixedPointType FixedPointType::get(::mlir::MLIRContext* ctxt, bool isSigned, unsigned whole, unsigned fractional) {
    return Base::get(ctxt, Types::FixedPoint, isSigned, whole, fractional);
}

FloatingPointType FloatingPointType::get(::mlir::MLIRContext* ctxt, bool isSigned, unsigned whole, unsigned fractional) {
    return Base::get(ctxt, Types::FloatingPoint, isSigned, whole, fractional);
}

template<typename FracTy>
Type fractionalTypeParse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
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

    return FracTy::get(ctxt, isSigned, whole, fractional);
}

Type FixedPointType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    return fractionalTypeParse<FixedPointType>(ctxt, parser);
}

Type FloatingPointType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    return fractionalTypeParse<FloatingPointType>(ctxt, parser);
}

void FixedPointType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getKeyword() << "<"
        << ( getImpl()->isSigned ? "true" : "false" ) << ","
        << getImpl()->whole << ","
        << getImpl()->fractional << ">";
}

void FloatingPointType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getKeyword() << "<"
        << ( getImpl()->isSigned ? "true" : "false" ) << ","
        << getImpl()->whole << ","
        << getImpl()->fractional << ">";
}

}
}
