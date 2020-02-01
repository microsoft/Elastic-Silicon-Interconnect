
@0x811a447312a70322;

struct HWType {
    union {
        bits @0 :UInt64; # Customize the number of bits used in hardware
        fixed :group { # Signed fixed point type
            whole @1 :UInt64; # Number of bits of whole
            fraction @2 :UInt64; # Number of bits of fraction
        }
    }
}