[Back to table of contents](index.md#Table-of-contents)

# ESI Type System: Basic Types

Here we present an overview of the base type system. Extensions to this to
support streaming or MMIO are described in those sections.

## Booleans, Integers, Fixed & Floating Point Numbers

**Keywords**: `bool`, `byte`, `bit`, `uint`, `int`, `fixed`, `float`, `ufixed`, `ufloat`

These types are all **parameterized** ([discussed
later](#parameterized-types)) by the number of bits they consume. For
instance, to get a 7-bit unsigned integer, use `uint<7>` or the `uint7`
alias. For a signed fixed point number with 2 bits of whole part and 10
bits of fraction (for a total of 13 bits, including the sign bit), use
`fixed<2, 10>`.

### Examples

| Name | Description |
| :--- | :--- |
| `float<10, 21>` | Signed floating point number -- 1 sign bit, 10 bits magnitude, and 21 bits mantissa |
| `uint9` | Unsigned integer -- 9 bits |
| `ufixed<0, 10>` | Unsigned fixed point -- 0 sign bits, 0 whole bits, 10 fraction bits |
| `int<100>` | Signed integer -- 1 sign bit, 100 whole bits |
| `fixed<0, 32>` | Signed fixed point -- 1 sign bit, 0 whole bits, 32 fraction bits |

Parameterized floating point and fixed point types do not have a natural
mapping into Verilog's type system. Language mappings (not described here)
define how non-native types are mapped.

## Enums

**Keyword**: `enum`

`Enums` specify a list of symbols which are automatically mapped to an
appropriately-sized `uint`, optionally, the specific numeric value of the
options. If specific values are not specified, the compiler assigns
values in-order, starting at 0 and skipping any values which have been
explicitly assigned. This is consistent with most software languages.

### Enum Example

```c++
enum Features {
    DDR, // Will get assigned 1
    Network = 0,
    PCIe, // Will get assigned 2
}
```

## Arrays

Any type can be made into an array of itself. Arrays must be statically
sized. Multidimensional arrays are supported but must be statically
sized in all dimensions. Note: in most languages by default, arrays will
be presented in one clock cycle so users are advised to use [Data
Windows](streaming.md#data-windows) on large arrays.

### Array Examples

| Name | Description | Total Size |
| - | - | - |
| `float<9, 22>[10]` | 10 x `float<9, 22>` | 320 bits |
| `uint9[12]` | 12 x `uint9` | 108 bits |
| `byte[9][4]` | 4 x 9 x `byte` | 288 bits |
| `ufixed<0, 10>[215]` | 215 x `ufixed<0, 10>` | 2150 bits |

## Structs

**Keyword**: `struct`

Structs are just like in C.

Elements in `structs` are expected to follow natural alignment in FPGA
memories and logic, and ABI alignment in CPU-memories. Alignment may be
specified for elements but will be respected only on CPUs and in [ESI
MMIO](mmio.md) regions. Alignment has no meaning on [streaming
channels](streaming.md). Packing is dependent on the context (size of
adjacent elements).

## Unions

**Keyword**: `union`, `c_union`

**`C_unions`** are like C unions, meaning that unions can be **any one**
of the types they contain but the type system does not know which.
Discriminated unions are also supported via **`union`** -- instances of unions
that implicitly include a tag specifying which of the members the instance
should be interpreted as.

### Union Examples

```c++
union GenericUnion {
    // This union is discriminated so it can be queried for the member
    // it represents
    struct {
        uint8 val;
    } data; // "data" is the tag symbol

    struct {} end; // "end" is the tag symbol
}
```

```c++
c_union DataStreamWithUnions {
    // Non-discriminated unions cannot be queried for tags, so that
    // information should be tracked separately
    struct {
        bool e;
        uint8 data;
    } data;

    struct {
        bool e; // Aliases with 'data.e' without a check
    } end;
}
```

## Lists

**Keyword**: list, fixed\_list

Lists are used to reason about variably-sized data. There are two types of
lists: those for which the size is known before transmission begins (a fixed
size list, or **fixed\_list**), and those for which it isn't (a variably
sized list, or just **list**). `Fixed_lists` are **very** roughly similar to
C++'s `std::vector` whereas `lists` are roughly similar to C++ `std::list`, a
linked list.

When lists are members of other data types (e.g. `structs`), they must be
completely read in the order in which they appear. A `list` is not
considered to be completely read until all items have been completely
read and the next members can be read. All `lists` must be completely read
before the outer (containing) message is considered read and the port
can move on to the next message.

### List Examples

```c++
list struct {
    fixed<4, 12> x;
    fixed<4, 12> y;
} CoordinateList;
```

```c++
struct EthernetFrame {
    uint64 Dst; // Absent a window, these fields are available
    uint64 Src; // until both lists are read
    uint32 VLan;
    uint16 Type;
    fixed_list byte Payload; // This list must be completely read
    list uint16 Second; // before this one.
    uint64 Footer; // This field is not accessible before
                   // the lists are completely read
}
```

## Parameterized Types

Any `struct` or `union` can include **parameters**. Parameters can be either
constant (compile time computable) integers or type names. These are like a
very simple version of C++ templates or C\# generics.

### Parameterized Types Examples

```c++
struct PrefixedList<type T> {
    uint10 header;
    fixed_list T data;
}
```

```c++
struct ArrayOf<type T, int N> {
    T[N] data;
}
```
