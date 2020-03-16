include ../capnp.mk

investData: stress1.capnp.txt stress1_fail.capnp.txt

run: investData
	../../SVCodeGen/bin/Debug/netcoreapp3.1/SVCodeGen stress1.capnp.bin

clean:
	rm -f *.txt *.bin
