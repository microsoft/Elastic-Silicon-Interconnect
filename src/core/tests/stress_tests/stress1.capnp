@0x94ed716c20b8728d;

using ESI = import "/EsiCoreAnnotations.capnp";

struct Polynomial3 { # ax^2 + bx + c
    a @0 :UInt32 $ESI.bits(24);
    b @1 :UInt32 $ESI.bits(40);
    # c @2 :Float32 $ESI.float(signed = true, exp = 3, mant = 10);
    c @2 :Float32;
}

struct Example {
    poly @0 :Polynomial3 $ESI.inline;

    exampleGroup :group {
        houseNumber @1 :UInt32;
        street @2 :Text;
        city @3 :Text;
        country @4 :Text;
    }

    subExample @5 :Example $ESI.inline;
}