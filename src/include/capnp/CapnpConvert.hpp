#ifndef __ESI_CAPNP_CONVERT__
#define __ESI_CAPNP_CONVERT__

#include <mlir/IR/Types.h>
#include <capnp/schema.h>
#include <vector>
#include <memory>

namespace esi {
namespace capnp {

    std::shared_ptr<std::vector<mlir::Type>> ConvertToESI(::capnp::Schema& s);
}
}
#endif
