//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "capnp/CapnpConvert.hpp"
#include "Dialects/Esi/EsiTypes.hpp"
#include "Dialects/Esi/EsiDialect.hpp"

#include "mlir/InitAllDialects.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <unistd.h>
#include <fcntl.h>

llvm::ExitOnError ExitOnError;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerDialect<mlir::esi::EsiDialect>();

  llvm::InitLLVM y(argc, argv);

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                  "ESI-Cap'nProto conversion utility\n");
  mlir::MLIRContext context;
  // mlir::esi::EsiDialect* esiDialect = new mlir::esi::EsiDialect(&context);

  // Read the CodeGeneratorRequest from the proper place
  int fd = 0;
  if (inputFilename != "-") {
    fd = open(inputFilename.c_str(), O_RDONLY);
  }

  capnp::ReaderOptions options;
  options.traversalLimitInWords = 1 << 30;  // Don't limit.
  capnp::StreamFdMessageReader reader(fd, options);
  auto request = reader.getRoot<capnp::schema::CodeGeneratorRequest>();
  if (fd != 0)
  {
    close(fd);
  }

  auto capnpVersion = request.getCapnpVersion();

  if (capnpVersion.getMajor() != CAPNP_VERSION_MAJOR ||
      capnpVersion.getMinor() != CAPNP_VERSION_MINOR ||
      capnpVersion.getMicro() != CAPNP_VERSION_MICRO) {
    auto compilerVersion = request.hasCapnpVersion()
        ? kj::str(capnpVersion.getMajor(), '.', capnpVersion.getMinor(), '.',
                  capnpVersion.getMicro())
        : kj::str("pre-0.6");  // pre-0.6 didn't send the version.
    auto generatorVersion = kj::str(
        CAPNP_VERSION_MAJOR, '.', CAPNP_VERSION_MINOR, '.', CAPNP_VERSION_MICRO);

    llvm::errs() <<
        "You appear to be using different versions of 'capnp' (the compiler) and "
        "'capnpc-c++' (the code generator). This can happen, for example, if you built "
        "a custom version of 'capnp' but then ran it with '-oc++', which invokes "
        "'capnpc-c++' from your PATH (i.e. the installed version). To specify an alternate "
        "'capnpc-c++' executable, try something like '-o/path/to/capnpc-c++' instead."
        << llvm::formatv("Expected version {}, got {}\n", compilerVersion.cStr(), generatorVersion.cStr());
  }

  // Open the output mlir assembly file
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // auto fp = mlir::esi::FloatingPointType::get(&context, true, 20, 4);
  // llvm::outs() << mlir::esi::ListType::get(&context, fp) << "\n";
  std::vector<mlir::Type> types;
  ExitOnError(esi::capnp::ConvertToESI(&context, request, types));
  // auto module = mlir::ModuleOp();
  // module.print(llvm::outs);
  for (auto type : types) {
    if (type == nullptr)
      continue;
    llvm::outs() << type.getKind() << ": " << type << "\n"; 
    // esiDialect->printType(t, nullptr);
  }

  llvm::outs() << "Success!\n";
  output->keep();
  return 0;
}
