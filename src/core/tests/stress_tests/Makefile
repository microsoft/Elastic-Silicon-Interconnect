investData:
	capnp compile -I../../schema/ -o- stress1.capnp > stress1.capnp.CodeGeneratorRequest.bin
	cat stress1.capnp.CodeGeneratorRequest.bin | capnp decode /usr/include/capnp/schema.capnp CodeGeneratorRequest > stress1.capnp.CodeGeneratorRequest.txt