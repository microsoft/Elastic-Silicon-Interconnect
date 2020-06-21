// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiTypes.hpp"
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Attributes.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>

using namespace mlir;

namespace mlir {
namespace esi {

namespace details {
    // ********************
    // Storage structs

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

    struct EmbeddedTypeStorage: public TypeStorage {
        EmbeddedTypeStorage(Type type)
            : TypeStorage(1), type(type) { }

        /// The hash key for this storage is a pair of the integer and type params.
        using KeyTy = Type;

        /// Define the comparison function for the key type.
        bool operator==(const KeyTy &key) const {
            return key == type;
        }

        static llvm::hash_code hashKey(const KeyTy &key) {
            return llvm::hash_combine(key);
        }

        /// Define a construction method for creating a new instance of this storage.
        static EmbeddedTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
            return new (allocator.allocate<EmbeddedTypeStorage>())
                EmbeddedTypeStorage(key);
        }

        Type type;
    };

    struct EmbeddedMultiTypeStorage: public TypeStorage {
        EmbeddedMultiTypeStorage(ArrayRef<MemberInfo> members)
            : members(members) { }

        /// The hash key for this storage is a pair of the integer and type params.
        using KeyTy = ArrayRef<MemberInfo>;

        /// Define the comparison function for the key type.
        bool operator==(const KeyTy &key) const {
            return key == members;
        }

        static llvm::hash_code hashKey(const KeyTy &key) {
            return llvm::hash_value(key);
        }

        /// Define a construction method for creating a new instance of this storage.
        static EmbeddedMultiTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
            llvm::ArrayRef<MemberInfo> members = allocator.copyInto(key);
            return new (allocator.allocate<EmbeddedMultiTypeStorage>())
                EmbeddedMultiTypeStorage(members);
        }

        ArrayRef<MemberInfo> members;
    };

    struct EnumTypeStorage: public TypeStorage {
        EnumTypeStorage(ArrayRef<std::string> members)
            : members(members) { }

        /// The hash key for this storage is a pair of the integer and type params.
        using KeyTy = ArrayRef<std::string>;

        /// Define the comparison function for the key type.
        bool operator==(const KeyTy &key) const {
            return key == members;
        }

        static llvm::hash_code hashKey(const KeyTy &key) {
            return llvm::hash_value(key);
        }

        /// Define a construction method for creating a new instance of this storage.
        static EnumTypeStorage *construct(TypeStorageAllocator &allocator,
                                            const KeyTy &key) {
            llvm::ArrayRef<std::string> members = allocator.copyInto(key);
            return new (allocator.allocate<EnumTypeStorage>())
                EnumTypeStorage(members);
        }

        ArrayRef<std::string> members;
    };
}

// ***********************
// Method bodies for fractional types

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

// *******************
// Method bodies for list class

ListType ListType::get(::mlir::MLIRContext* ctxt, Type type) {
    return Base::get(ctxt, Types::List, type);
}

Type ListType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    Type type;
    if (parser.parseLess()) return Type();
    parser.parseType(type);
    if (parser.parseGreater()) return Type();
    return get(ctxt, type);
}

void ListType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getKeyword() << "<";
    printer.printType(getImpl()->type);
    printer << ">";
}

mlir::Type ListType::getContainedType() {
    return getImpl()->type;
}

// ****************
// Method bodies for compound types

StructType StructType::get(
        ::mlir::MLIRContext* ctxt,
        ArrayRef<MemberInfo> members) {
    return Base::get(ctxt, Types::Struct, members);
}
UnionType UnionType::get(
        ::mlir::MLIRContext* ctxt,
        ArrayRef<MemberInfo> members) {
    return Base::get(ctxt, Types::Union, members);
}

ParseResult parseMember(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser, MemberInfo& member) {
    if (parser.parseLBrace()) return mlir::failure();
    if (parser.parseType(member.type)) return mlir::failure();
    if (succeeded(parser.parseOptionalComma())) {
        StringAttr a;
        if (parser.parseAttribute<StringAttr>(a)) return mlir::failure();
        member.name = a.getValue().str();
    }
    if (parser.parseRBrace()) return mlir::failure();
    return success();
}

template<typename ThisType>
Type parseCompound(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    SmallVector<MemberInfo, 1> members;
    if (parser.parseLess()) return Type();
    do {
        MemberInfo member;
        parseMember(ctxt, parser, member);
        members.push_back(member);
    } while(succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater()) return Type();
    return ThisType::get(ctxt, members);
}

Type StructType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    return parseCompound<StructType>(ctxt, parser);
}

Type UnionType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    return parseCompound<UnionType>(ctxt, parser);
}

void printMember(mlir::DialectAsmPrinter& printer, const MemberInfo& member) {
    printer << "{";
    printer.printType(member.type);
    if (member.name.size() > 0) {
        printer << ",\"" << member.name << "\"";
    }
    printer << "}";
}

void printCompound(mlir::DialectAsmPrinter& printer, StringRef keyword, ArrayRef<MemberInfo> members) {
    printer << keyword << "<";
    for (size_t i=0; i<members.size(); i++) {
        printMember(printer, members[i]);
        if (i + 1 < members.size()) {
            printer << ",";
        }
    }
    printer << ">";
}

void StructType::print(mlir::DialectAsmPrinter& printer) const {
    printCompound(printer, getKeyword(), getImpl()->members);
}

void UnionType::print(mlir::DialectAsmPrinter& printer) const {
    printCompound(printer, getKeyword(), getImpl()->members);
}

// ****************
// Enum bodies

EnumType EnumType::get(::mlir::MLIRContext* ctxt, ArrayRef<std::string> members) {
    return Base::get(ctxt, Types::Enum, members);
}

Type EnumType::parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser) {
    SmallVector<std::string, 1> members;
    if (parser.parseLess()) return Type();
    do {
        StringAttr sa;
        if (parser.parseAttribute<StringAttr>(sa)) return Type();
        members.push_back(sa.getValue().str());
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater()) return Type();
    return get(ctxt, members);
}

void EnumType::print(mlir::DialectAsmPrinter& printer) const {
    printer << getKeyword() << "<";
    auto members = getImpl()->members;
    for (size_t i=0; i<members.size(); i++) {
        printer << "\"" << members[i] << "\"";
        if (i + 1 < members.size()) {
            printer << ",";
        }
    }
    printer << ">";
}

}
}
