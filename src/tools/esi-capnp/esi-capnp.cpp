//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "capnp/CapnpConvert.hpp"
#include "Dialects/Esi/EsiDialect.hpp"

#include "mlir/InitAllDialects.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include "capnp/schema-parser.h"

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

  // Get the path to the current executable since that's where the Capnp header is
  auto fs = kj::newDiskFilesystem();
  auto exePathStr = fs->getRoot().readlink(kj::Path::parse("proc/self/exe"));
  kj::StringPtr exePathRelRoot = exePathStr.cStr()+1;
  auto exeDir = fs->getRoot().openSubdir(kj::Path::parse(exePathRelRoot).parent());

  // Load the input schema
  capnp::SchemaParser parser;
  auto schema = parser.parseFromDirectory(fs->getCurrent(),
      kj::Path::parse(inputFilename), { exeDir });

  // Open the output mlir assembly file
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  std::vector<mlir::Type> types;
  esi::capnp::ConvertToESI(schema, types);

  llvm::outs() << "Success!\n";
  output->keep();
  return 0;
}
