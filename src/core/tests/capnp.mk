.SUFFIXES: .capnp .capnp.txt .capnp.bin

.capnp.bin.capnp.txt:
	cat $*.capnp.bin | capnp decode /usr/include/capnp/schema.capnp CodeGeneratorRequest > $*.capnp.txt

.capnp.capnp.bin:
	capnp compile -I../../schema/ -o- $*.capnp > $*.capnp.bin