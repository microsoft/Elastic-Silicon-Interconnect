@0x811a447312a70000;

##############
## Elastic Silicon Interconnect hardware annotations
##############

# Customize the number of bits used in hardware
annotation bits @0xac112269228ad38c (struct, union, field, enum) :UInt64; 

# Mark field as inline in hardware
annotation inline @0x83f1b26b0188c1bb (field) :Void;

# A fixed-size array of certain length (inline implied)
annotation array @0x93ce43d5fd6478ee (field) :UInt64;

# A non-discriminated union
annotation cUnion @0xed2e4e8a596d00a5 (union) :Void;

# A variably sized list wherein the length is know before transmission
annotation fixedList @0x8e0d4f6349687e9b (field) :Void;

# Fixed point type
annotation fixed @0xb0aef92d8eed92a5 (field) :FixedPointSpec; 
struct FixedPointSpec @0x82adb6b7cba4ca97 {
    signed @0 :Bool;
    whole @1 :UInt64; # Number of bits of whole
    fraction @2 :UInt64; # Number of bits of fraction
}

# If desired, a user can instantiate this struct to obtain a lossless value
# To customize the bitwidth in hardware, use the $fixed attribute
struct FixedPointValue @0x81eebdd3a9e24c9d {
    whole @0 :Int64; # The whole part of the value
    fraction @1 :UInt64; # The fractional part of the value
}

# Flexible floating point number
annotation float @0xc06dd6e3ee4392de (field) :FloatingPointSpec;
struct FloatingPointSpec @0xa9e717a24fd51f71 { 
    signed @0 :Bool;
    exp @1 :UInt64; # Number of bits of exponent
    mant @2 :UInt64; # Number of bits mantissa
}

# If desired, a user can instantiate this struct to obtain a lossless value
# To customize the bitwidth in hardware, use the $float attribute
struct FloatingPointValue @0xaf862f0ea103797c {
    exp @0 :Int64; # The exponent and sign bit
    mant @1 :UInt64; # The mantissa
}

# Make field accessible with this offset in MMap'ed structs
annotation offset @0xcdbc3408a9217752 (field) :HWOffset;
struct HWOffset @0xf7afdfd9eb5a7d15 {
    #union {
        #bytes @0 :UInt64;
        #bit @1 :UInt64;
    #}
}

