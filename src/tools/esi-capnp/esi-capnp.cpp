//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/Esi/EsiDialect.hpp"
#include "Dialects/Esi/EsiTypes.hpp"
#include "capnp/CapnpConvert.hpp"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#include <capnp/message.h>
#include <capnp/schema-parser.h>
#include <capnp/serialize.h>

#include <fcntl.h>
#include <unistd.h>

#include <map>

llvm::ExitOnError ExitOnError;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::ValueRequired,
                                                llvm::cl::desc("<input file>"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  mlir::enableGlobalDialectRegistry(true);
  mlir::registerAllDialects();
  mlir::registerDialect<mlir::esi::EsiDialect>();

  llvm::InitLLVM y(argc, argv);

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ESI-Cap'nProto conversion utility\n");
  mlir::MLIRContext context;

  kj::Own<kj::Filesystem> fs = kj::newDiskFilesystem();
  if (inputFilename.size() == 0) {
    llvm::errs() << "Input filename must specified\n";
    return 1;
  }

  // for (auto a : esi::capnp::Annotations::idToName) {
  //   llvm::outs() << llvm::formatv("id: {0} = {1}\n", a.first, a.second);
  // }

  // Set up paths to parse the damn schema
  kj::Path pathEvaler(nullptr);
  kj::Path inputFilenameAbs =
      (*inputFilename.begin() == '/')
          ? pathEvaler.evalNative(inputFilename)
          : fs->getCurrentPath().append(kj::Path::parse(inputFilename));

  auto baseDir = fs->getRoot().openSubdir(inputFilenameAbs.parent());
  const auto exeFile = fs->getRoot().readlink(kj::Path::parse("proc/self/exe"));
  auto exeDir =
      fs->getRoot().openSubdir(pathEvaler.evalNative(exeFile).parent());

  if (!fs->getRoot().exists(inputFilenameAbs)) {
    llvm::errs() << "Input file does not exist!\n";
    return 1;
  }

  // Parse the damn thing
  capnp::SchemaParser parser;
  auto rootSchema = parser.parseFromDirectory(
      *baseDir, inputFilenameAbs.basename().clone(), {exeDir.get()});

  // Open the output mlir assembly file
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Do the conversion
  std::map<std::string, mlir::Type> types;
  ExitOnError(esi::capnp::ConvertToESI(&context, rootSchema, types));

  // Print the output
  for (auto type : types) {
    if (type.second == nullptr)
      continue;
    llvm::outs() << "\n";
    llvm::outs() << llvm::raw_ostream::CYAN << type.first << ": "
                 << llvm::raw_ostream::GREEN << type.second << "\n";
    llvm::outs().resetColor();
  }

  llvm::outs() << "Success!\n";
  output->keep();
  return 0;
}
