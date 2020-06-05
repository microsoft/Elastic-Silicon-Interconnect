<!--
  Copyright (c) Microsoft Corporation.
  Licensed under the MIT License.
-->
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
    API["ESI typed<br/>CapnProto API"]
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

## ESI-annotated CapnProto schema to normal Capnp schema

```mermaid
graph LR;
    EC["ESI-annotated Capnp schema"]
    EsiConv["ESI schema model"]
    CS["ESI-generated Capnp schema"]
    EC --> EsiConv --> CS
```

Basic idea: write a Capnp-schema to Capnp-schema translator which goes
through the ESI Type Schema. It would necessarily be lossy.
