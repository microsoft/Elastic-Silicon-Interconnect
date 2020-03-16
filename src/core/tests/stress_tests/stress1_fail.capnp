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

struct Shape {
    area @0 :Float64 $ESI.array(5);

    #union {
        circle @1 :Float64;      # radius
        square @2 :Float64;      # width
    #}

    volume @3 :Float64 $ESI.fixedList;

    #volumeShape :union {
        cube :group {
            length @4 :ESI.FixedPointValue;
            width @5 :ESI.FixedPointValue
                $ESI.fixed(
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

struct Map(Key, Value) {
  entries @0 :List(Entry);
  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}

struct UnionTest
{
    u :union {
        a @0 :Void;
        b @1 :Void;
    }
}

interface Foo
{ }

struct InterfaceTest
{
    i @0 :Foo;
    p @1 :AnyPointer;
}
