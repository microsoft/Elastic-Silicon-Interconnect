.SUFFIXES: .capnp .capnp.txt .capnp.bin

.capnp.bin.capnp.txt:
ifeq ($(OS), Windows_NT)
	capnp decode ../../capnp/CapnpSchema.capnp CodeGeneratorRequest < $*.capnp.bin  > $*.capnp.txt
else
	cat $*.capnp.bin | capnp decode ../../capnp/CapnpSchema.capnp CodeGeneratorRequest > $*.capnp.txt
endif

.capnp.capnp.bin:
	capnp compile -I../../capnp/ -o- $*.capnp > $*.capnp.bin