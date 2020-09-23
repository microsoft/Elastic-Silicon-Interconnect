#ifndef __ESI_CAPNP_CONVERT__
#define __ESI_CAPNP_CONVERT__

#include <capnp/schema-parser.h>
#include <capnp/schema.h>
#include <llvm/Support/Error.h>
#include <map>
#include <memory>
#include <mlir/IR/Types.h>
#include <vector>

namespace esi {
namespace capnp {

llvm::Error ConvertToESI(::mlir::MLIRContext *,
                         ::capnp::ParsedSchema &rootSchema,
                         ::std::map<::std::string, mlir::Type> &outputTypes);

}
} // namespace esi
#endif
