# High Level Interfaces: Concepts

This set of documents provides a draft of the fundamental HLI concepts. Early
implementations may not have all the functionality discussed herein.

## Terminology

Module interfaces are defined by a set of **typed ports** over which
**messages** are exchanged. **Message types** include parameterized sized
**integer**s, **fixed** point numbers, **float**ing point numbers, fixed size
**arrays**, **enum**s, **union**s, **structs**, and variably-sized **lists**.
Additionally, **data windows** can be specified on messages, allowing modules
to specify a narrower port than the message type. Streaming connections
between **HLI ports** are called **channels**. Modules with only High-Level
Interface (HLI) ports are called **HLI modules**. Modules with mixed
wire-level ports and HLI ports are allowed, but functionality which requires
HLI ports may not function on these modules. Modules can expose **HLI MMIO**
regions for memory-mapped I/O.

It is important to note that the message type does not specify anything
pertaining to the wire/cycle level. A large struct may be serialized across
several clock cycles and re-assembled in the receiver. Structs and other
atomic data types (everything except lists and data windows), however, are
guaranteed to be complete before they are handed to receivers.

## A note on syntax

This document contains pseudocode examples mostly written in a C-style. The
examples are intended to be demonstrative of concepts, and thus the syntax is
irrelevant. When we go about describing HLI as an extension to SystemVerilog
and other languages, we expect the syntax to be quite different, but the
semantics to be identical.

## Table of contents

1) [Base Type System](types.md)
1) [Streaming Channels](streaming.md)
1) [Memory-Mapped Interfaces](mmio.md)
1) [Software APIs](software_api.md)
1) [Services](services.md)
1) [Misc. Notes](notes.md)
