
@0x811a447312a70322;

annotation hw (struct, union, field, enum) :HWType;

struct HWType @0xe407187a197a1bc1 {
    union {
        bits @0 :UInt64; # Customize the number of bits used in hardware
        fixed :group { # Signed fixed point type
            whole @1 :UInt64; # Number of bits of whole
            fraction @2 :UInt64; # Number of bits of fraction
        }
        float :group {
            exp @3 :UInt64;
            mant @4 :UInt64;
        }
        cUnion @5 :Void;
    }
}