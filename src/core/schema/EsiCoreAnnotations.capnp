@0x811a447312a70322;

##############
## Elastic Silicon Interconnect hardware annotations
##############

# Customize the number of bits used in hardware
annotation bits(struct, union, field, enum) :UInt64; 

# Mark field as inline in hardware
annotation inline (field) :Void;

# A fixed-size array of certain length (inline implied)
annotation array (field) :UInt64;

# A non-discriminated union
annotation cUnion (union) :Void;

# A variably sized list wherein the length is know before transmission
annotation fixedList (field) :Void;

# Fixed point type
annotation fixed (field) :FixedPoint; 
struct FixedPoint { 
    signed @0 :Bool;
    whole @0 :UInt64; # Number of bits of whole
    fraction @1 :UInt64; # Number of bits of fraction
}

# Flexible floating point number
annotation float (field) :Float;
struct Float { 
    signed @0 :Bool;
    exp @1 :UInt64; # Number of bits of exponent
    mant @2 :UInt64; # Number of bits mantissa
}

# Make field accessible with this offset in MMap'ed structs
annotation offset (field) :HWOffset;
struct HWOffset @0xf7afdfd9eb5a7d15 {
    union {
        bytes @0 :UInt64;
        bit @1 :UInt64;
    }
}

