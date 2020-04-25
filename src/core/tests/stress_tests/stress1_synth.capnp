@0x94ed716c20b8728d;

using ESI = import "/EsiCoreAnnotations.capnp";

struct Example {
    poly @0 :Polynomial3 $ESI.inline;

    exampleGroup :group {
        houseNumber @1 :UInt32;
        apt @3 :UInt32;
        #street @2 :Text;
        #city @3 :Text $ESI.inline;
    }

    after @2 :UInt16;
}

struct Polynomial3 { # ax^2 + bx + c
    a @0 :Int32 $ESI.bits(24);
    b @1 :UInt32 $ESI.bits(40);
        c @2 :Float32 $ESI.float(signed = true, exp = 3, mant = 10);
        d @3 :Float32 $ESI.float(signed = true, exp = 3, mant = 10);
    g :group { 
        e @4 :Float32 $ESI.float(signed = true, exp = 3, mant = 10);
    }
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

    array1 @8 :List(UInt32) $ESI.bits(24) $ESI.array(3);
}

struct All {
    a @0 :Void;
    b @1 :Bool;
    c @2 :Int8;
    d @3 :Int16;
    e @4 :Int32;
    f @5 :Int64;
    g @6 :UInt8;
    h @7 :UInt16;
    i @8 :UInt32;
    j @9 :UInt64;
    k @10 :Float32;
    l @11 :Float64;
}

interface ShapeQuery {
    getVolume @0 (Shape :Shape) -> (Volume :Float32);
}

interface Polynomial3Compute {
    compute @0 (coeff :Polynomial3, x :Float32) -> (y :Float32);
    solveForX @1 (coeff :Polynomial3, y :UInt16) -> (x1 :Float32, x2 :Float32);
}
