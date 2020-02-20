investData:
	capnp compile -I../../schema/ -o- toy.capnp > toy.capnp.CodeGeneratorRequest.bin
	cat toy.capnp.CodeGeneratorRequest.bin | capnp decode /usr/include/capnp/schema.capnp CodeGeneratorRequest > toy.capnp.CodeGeneratorRequest.txt