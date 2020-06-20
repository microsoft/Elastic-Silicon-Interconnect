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
#include <capnp/schema-parser.h>

#include <unistd.h>
#include <fcntl.h>

llvm::ExitOnError ExitOnError;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"));
                                                

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

  kj::Own<kj::Filesystem> fs = kj::newDiskFilesystem();
  auto inputPath = kj::Path::parse(inputFilename.getValue());
  auto& cwd = fs->getCurrent();
  auto baseDir = cwd.openSubdir(inputPath.parent());
  auto importPaths = kj::ArrayPtr<const kj::ReadableDirectory*> {

  };

  capnp::SchemaParser parser;
  auto rootSchema = parser.parseFromDirectory(
    *baseDir,
    kj::Path::parse(inputFilename.getValue()),
    importPaths);

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
  // ExitOnError(esi::capnp::ConvertToESI(&context, request, types));
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
