#!/bin/bash

mkdir mlir/build
cd mlir/build
cmake -G Ninja ../llvm \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_BUILD_EXAMPLES=ON \
	-DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
	-DCMAKE_BUILD_TYPE=Debug \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DLLVM_INSTALL_UTILS=ON \
	-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON \
	-DBUILD_SHARED_LIBS=ON
	# Building shared libs really speeds up linking

cmake --build . --target check-mlir
