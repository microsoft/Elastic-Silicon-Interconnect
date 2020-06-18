#ifndef __ESI_CAPNP_CONVERT__
#define __ESI_CAPNP_CONVERT__

#include <mlir/IR/Types.h>
#include <llvm/Support/Error.h>
#include <capnp/schema.h>
#include <capnp/schema-parser.h>
#include <vector>
#include <memory>

namespace esi {
namespace capnp {
    llvm::Error ConvertToESI(
        ::mlir::MLIRContext*,
        ::capnp::schema::CodeGeneratorRequest::Reader& cgr,
        ::std::vector<mlir::Type>& outputTypes);
}
}
#endif
