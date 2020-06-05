// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Operation.h>

namespace mlir {
namespace esi {

enum Types {
    Compound = Type::FIRST_PRIVATE_EXPERIMENTAL_6_TYPE,
};

struct CompoundTypeStorage : public TypeStorage {
    CompoundTypeStorage(bool isSigned, unsigned whole, unsigned fractional)
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
    static CompoundTypeStorage *construct(TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
        auto [isSigned, whole, fractional] = key;
        return new (allocator.allocate<CompoundTypeStorage>())
            CompoundTypeStorage(isSigned, whole, fractional);
    }

    bool isSigned;
    unsigned whole;
    unsigned fractional;
};


/// This class defines a parametric type. All derived types must inherit from
/// the CRTP class 'Type::TypeBase'. It takes as template parameters the
/// concrete type (ComplexType), the base class to use (Type), and the storage
/// class (ComplexTypeStorage). 'Type::TypeBase' also provides several utility
/// methods to simplify type construction and verification.
class CompoundType : public Type::TypeBase<CompoundType, Type,
                                        CompoundTypeStorage> {
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// This static method is used to support type inquiry through isa, cast,
    /// and dyn_cast.
    static bool kindof(unsigned kind) { return kind == Types::Compound; }

    /// This method is used to get an instance of the 'ComplexType'. This method
    /// asserts that all of the construction invariants were satisfied. To
    /// gracefully handle failed construction, getChecked should be used instead.
    static CompoundType get(::mlir::MLIRContext* ctxt, bool isSigned, unsigned whole, unsigned fractional) {
        // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
        // of this type. All parameters to the storage class are passed after the
        // type kind.
        return Base::get(ctxt, Types::Compound, isSigned, whole, fractional);
    }

    /// This method is used to verify the construction invariants passed into the
    /// 'get' and 'getChecked' methods. Note: This method is completely optional.
    static LogicalResult verifyConstructionInvariants(
        Location loc, bool isSigned, unsigned whole, unsigned fractional) {
        if (fractional == 0)
            return ::mlir::emitError(loc) << "fractional part of 'CompoundType' cannot be zero";
        return success();
    }

    /// Return the parameter value.
    // unsigned getParameter() {
    //     // 'getImpl' returns a pointer to our internal storage instance.
    //     return getImpl()->nonZeroParam;
    // }

    // /// Return the integer parameter type.
    // IntegerType getParameterType() {
    //     // 'getImpl' returns a pointer to our internal storage instance.
    //     return getImpl()->integerType;
    // }
};

}
}

#endif
