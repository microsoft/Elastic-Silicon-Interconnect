# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# include $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/../capnp.mk
include ../capnp.mk


investData: stress1_synth.capnp.txt stress1.capnp.txt stress1_fail.capnp.txt

run: stress1_synth.capnp.bin
	dotnet run --project ../../SVCodeGen -- -i stress1_synth.capnp.bin

write:
	-dotnet test ../../ --filter name=ReadWriteCompareStress1Synth
	capnp compile -I../../capnp.convert/ -o- schema1_synth.esi.capnp > schema1_synth.esi.capnp.bin
	capnp decode ../../capnp.convert/CapnpSchema.capnp CodeGeneratorRequest < schema1_synth.esi.capnp.bin  > schema1_synth.esi.capnp.txt

clean:
	rm -f *.txt *.bin *.esi.svh *.esi.sv
	rm -rf obj_dir work __pycache__

verilator: run
ifeq ($(OS), Windows_NT)
	verilator_bin --cc -Wno-fatal *.sv --top-module Shape_tb
else
	verilator --cc -Wno-fatal *.sv --top-module Shape_tb
endif

vlog: run
	vlog -sv09compat *.sv
	vopt Shape_tb -o Shape_tb_vopt

vsim: vlog
	vsim Shape_tb_vopt -c -do "run 100; exit"
