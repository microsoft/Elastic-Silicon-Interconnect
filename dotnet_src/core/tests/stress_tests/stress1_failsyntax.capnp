@0x94ed716c20b8728d;

using ESI = import "/EsiCoreAnnotations.capnp";

struct Example {
    poly @0 :Polynomial3 $ESI.inline;

    exampleGroup :group {
        houseNumber @1 :UInt32;
        street @2 :Text;
        city @3 :Text $ESI.inline;
    }

    subExample @4 :Example $ESI.inline;
}

struct Polynomial3 { # ax^2 + bx + c
    a @0 :UInt32 $ESI.bits(24);
    b @1 :UInt32 $ESI.float(signed = true, exp = 3, mant = 10);
    c @2 :Float32 $ESI.bits(40);
}


                    signed = false,
                    whole = 4,
                    fraction = 10);
            height @6 :UInt32
                $ESI.float(
                    signed = false,
                    exp = 4,
                    mant = 2);
        }
        sphere @7 :Bool $ESI.fixed(signed = false, whole = 4, fraction = 12);
    #}
}

interface Foo
{ }
