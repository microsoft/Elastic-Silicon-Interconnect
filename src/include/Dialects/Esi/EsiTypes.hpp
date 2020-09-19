// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>

namespace mlir {
namespace esi {

  struct FieldInfo
  {
    StringRef name;
    Type type;

  public:
    FieldInfo(const FieldInfo &fi) : name(fi.name), type(fi.type) {}
    FieldInfo(StringRef name, Type type) : name(name), type(type) {}

    FieldInfo allocateInto(TypeStorageAllocator& alloc) const {
      return FieldInfo(alloc.copyInto(name), type);
    }
  };

  static bool operator==(const FieldInfo& a, const FieldInfo& b) {
    return a.name == b.name && a.type == b.type;
  }

  static llvm::hash_code hash_value(const FieldInfo& fi) {
    return llvm::hash_combine(fi.name, fi.type);
  }

} // namespace esi
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Dialects/Esi/EsiDialectTypes.h.inc"

#endif
