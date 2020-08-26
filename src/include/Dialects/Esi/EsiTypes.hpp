// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_TYPES_HPP__
#define __ESI_TYPES_HPP__

#include <tuple>

#include <mlir/IR/Dialect.h>
#include <mlir/IR/Operation.h>

namespace mlir {
namespace esi {

  struct FieldInfo {
  public:
    StringRef name;
    Type type;

    FieldInfo allocateInto(TypeStorageAllocator& alloc) const {
      return FieldInfo {
        .name = alloc.copyInto(name),
        .type = type
      };
    }
  };

  static bool operator==(const FieldInfo& a, const FieldInfo& b) {
    return a.name == b.name && a.type == b.type;
  }

  static llvm::hash_code hash_value(const FieldInfo& fi) {
    return llvm::hash_combine(fi.name, fi.type);
  }

#define GET_TYPEDEF_CLASSES
#include "Dialects/Esi/EsiDialectTypes.h.inc"


}
}

#endif
