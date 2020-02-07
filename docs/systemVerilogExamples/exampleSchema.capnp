@0x94ed716c20b8728d;

using ESI = import "/EsiCoreAnnotations.capnp";

struct Polynomial3 { # ax^2 + bx + c
    a @0 :UInt32 $ESI.bits(24);
    b @1 :UInt32 $ESI.bits(20);
    c @2 :UInt16;
}

