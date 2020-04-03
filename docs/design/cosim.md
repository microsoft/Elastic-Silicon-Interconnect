# ESI Co-simulation design doc

## Core cosim.dll

```mermaid
graph LR;
    subgraph cosim.dll
        RS[Capnp RPC Server]
        USwQ["Software queues<br/>(untyped)"]
        RS --> USwQ
    end
    subgraph "RTL Simulator"
        URQ["Untyped RTL<br/>DPI-C $functions"]
        TRQ[Typed RTL queues]
        URQ --> TRQ
    end
    USwQ -- DPI polling calls --> URQ
```

This CapnProto RPC server has a very simple interface: an enumeration
interface and a number of blob send/recv channels. It is untyped with regard
to the ESI schema.

## ESI C++ interface

```mermaid
graph TB;
    C[Client code]
    API["ESI typed<br/>(generated) API"]
    UnAPI["ESI hardware serializer"]
    C --> API
    API --> UnAPI

    CS["Cosim backend"]
    P["Hardware PCIe backends"]
    N["Hardware network backends"]
    D["..."]
    UnAPI --> CS
    UnAPI --> P
    UnAPI --> N
    UnAPI --> D
```

The ESI-generated C++ API has several backend plugins, one of which is the
cosim backend which talks to a cosim server.

## CapnProto RPC interface

```mermaid
graph LR;
    RPC["CapnProto RPC server C++ API"]
    TypeConv["CapnProto to ESI C struct converter"]
    FuncMap["RPC to ESI function mapping"]
    API["ESI typed API"]

    RPC --> TypeConv
    RPC --> FuncMap
    FuncMap --> API
    TypeConv --> API
```

The CapnProto RPC interface is simply a shim to the ESI generated API. It can
thus act as a proxy to cosim or hardware.
