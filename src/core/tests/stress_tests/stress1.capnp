@0x94ed716c20b8728d;

using ESI = import "/EsiCoreAnnotations.capnp";

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

struct Polynomial3 { # ax^2 + bx + c
    a @0 :UInt32 $ESI.bits(24);
    b @1 :UInt32 $ESI.bits(40);
    c @2 :Float32 $ESI.float(signed = true, exp = 3, mant = 10);
}

struct Shape {
    area @0 :Float64;

    #union {
        circle @1 :Float64;      # radius
        square @2 :Float64;      # width
    #}

    volume @3 :Float64;

    #volumeShape :union {
        cube :group {
            length @4 :ESI.FixedPointValue;
            width @5 :ESI.FixedPointValue
                $ESI.fixed(
                    signed = false,
                    whole = 4,
                    fraction = 10);
            height @6 :ESI.FloatingPointValue
                $ESI.float(
                    signed = false,
                    exp = 4,
                    mant = 2);
        }
        sphere @7 :Float32 $ESI.fixed(signed = false, whole = 4, fraction = 12);
    #}
}
