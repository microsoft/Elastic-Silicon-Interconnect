include ../capnp.mk

investData: stress1.capnp.txt stress1_fail.capnp.txt

run: stress1_synth.capnp.bin
	dotnet run --project ../../SVCodeGen -- stress1_synth.capnp.bin

clean:
	rm -f *.txt *.bin

vlog: run
	vlog -sv09compat *.sv

shape: vlog
	vopt Shape_tb -o Shape_tb_vopt
	vsim Shape_tb_vopt -c -do "run 100; exit"
