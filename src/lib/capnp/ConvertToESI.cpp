// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "capnp/CapnpConvert.hpp"
#include "Dialects/Esi/EsiTypes.hpp"

#include <vector>
#include <memory>

using namespace std;
using namespace mlir::esi;

namespace esi {
namespace capnp {

shared_ptr<vector<mlir::Type>> ConvertToESI(::capnp::Schema& s) {
    auto types = make_shared<vector<mlir::Type>>();
    return types;
}

}
}
