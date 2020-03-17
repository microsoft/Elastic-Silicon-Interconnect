.SUFFIXES: .capnp .capnp.txt .capnp.bin

.capnp.bin.capnp.txt:
ifeq ($(OS), Windows_NT)
	gci $*.capnp.bin | capnp decode ../../schema/CapnpSchema.capnp CodeGeneratorRequest > $*.capnp.txt
else
	$*.capnp.bin | capnp decode ../../schema/CapnpSchema.capnp CodeGeneratorRequest > $*.capnp.txt
endif

.capnp.capnp.bin:
	capnp compile -I../../schema/ -o- $*.capnp > $*.capnp.bin