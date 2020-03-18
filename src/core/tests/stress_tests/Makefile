include ../capnp.mk

investData: stress1.capnp.txt stress1_fail.capnp.txt

run: stress1_synth.capnp.bin
	../../SVCodeGen/bin/Debug/netcoreapp3.1/SVCodeGen stress1_synth.capnp.bin

clean:
	rm -f *.txt *.bin

vlog: run
	vlog -sv09compat *.sv
