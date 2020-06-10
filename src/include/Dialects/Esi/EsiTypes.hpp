// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>

namespace mlir {
namespace esi {

namespace details {
    struct FractionalTypeStorage;
    struct EmbeddedTypeStorage;
    struct EmbeddedMultiTypeStorage;
}

enum Types {
    FixedPoint = Type::FIRST_PRIVATE_EXPERIMENTAL_6_TYPE,
    FloatingPoint,
    List,
    Struct,
    Union
};

class FixedPointType : public Type::TypeBase<FixedPointType, Type,
                                        details::FractionalTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == Types::FixedPoint; }

    static StringRef getKeyword() { return "fixed"; }

    static FixedPointType get(::mlir::MLIRContext* ctxt, bool isSigned, unsigned whole, unsigned fractional);

    static LogicalResult verifyConstructionInvariants(
        Location loc, bool isSigned, unsigned whole, unsigned fractional) {
        if (fractional == 0)
            return ::mlir::emitError(loc) << "fractional part of fixed point number cannot be zero width";
        return success();
    }

    static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
    void print(mlir::DialectAsmPrinter& printer) const;
};

class FloatingPointType : public Type::TypeBase<FloatingPointType, Type,
                                        details::FractionalTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == Types::FloatingPoint; }

    static StringRef getKeyword() { return "float"; }

    static FloatingPointType get(::mlir::MLIRContext* ctxt, bool isSigned, unsigned whole, unsigned fractional);

    static LogicalResult verifyConstructionInvariants(
        Location loc, bool isSigned, unsigned exp, unsigned mantissa) {
        if (exp == 0)
            return ::mlir::emitError(loc) << "exponent part of floating point number cannot be zero width";
        return success();
    }

    static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
    void print(mlir::DialectAsmPrinter& printer) const;
};

class ListType : public Type::TypeBase<ListType, Type,
                                        details::EmbeddedTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == Types::List; }

    static StringRef getKeyword() { return "list"; }

    static ListType get(::mlir::MLIRContext* ctxt, Type);

    // static LogicalResult verifyConstructionInvariants(
    //     Location loc, bool isSigned, unsigned whole, unsigned fractional) {
    //     if (fractional == 0)
    //         return ::mlir::emitError(loc) << "fractional part of fixed point number cannot be zero width";
    //     return success();
    // }

    static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
    void print(mlir::DialectAsmPrinter& printer) const;
};

struct MemberInfo {
public:
    StringRef name;
    ::mlir::Type type;
};

inline bool operator==(const MemberInfo& LHS, const MemberInfo& RHS) {
    return
        LHS.name == RHS.name &&
        LHS.type == RHS.type;
}

inline llvm::hash_code hash_value(const MemberInfo& member) {
    return llvm::hash_combine(member.name, member.type);
}

class StructType : public Type::TypeBase<StructType, Type,
                                        details::EmbeddedMultiTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == Types::Struct; }

    static StringRef getKeyword() { return "struct"; }

    static StructType get(::mlir::MLIRContext* ctxt, llvm::ArrayRef<MemberInfo> members);

    // static LogicalResult verifyConstructionInvariants(
    //     Location loc, bool isSigned, unsigned whole, unsigned fractional) {
    //     if (fractional == 0)
    //         return ::mlir::emitError(loc) << "fractional part of fixed point number cannot be zero width";
    //     return success();
    // }

    static Type parse(mlir::MLIRContext* ctxt, mlir::DialectAsmParser& parser);
    void print(mlir::DialectAsmPrinter& printer) const;
};


}
}

#endif
