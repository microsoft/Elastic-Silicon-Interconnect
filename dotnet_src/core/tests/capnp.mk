.SUFFIXES: .capnp .capnp.txt .capnp.bin

.capnp.bin.capnp.txt:
ifeq ($(OS), Windows_NT)
	capnp decode ../../capnp.convert/CapnpSchema.capnp CodeGeneratorRequest < $*.capnp.bin  > $*.capnp.txt
else
	cat $*.capnp.bin | capnp decode ../../capnp.convert/CapnpSchema.capnp CodeGeneratorRequest > $*.capnp.txt
endif

.capnp.capnp.bin:
	capnp compile -I../../capnp.convert/ -o- $*.capnp > $*.capnp.bin