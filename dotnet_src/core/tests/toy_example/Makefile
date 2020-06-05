# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

include ../capnp.mk

investData: toy.capnp.txt

run: toy.capnp.bin
	../../SVCodeGen/bin/Debug/netcoreapp3.1/SVCodeGen.exe toy.capnp.bin

vericompile: run
	verilator --lint-only *.sv

vlog: run
	vlog -sv09compat *.sv

vsim: vlog
	vsim Polynomial3Compute_tb