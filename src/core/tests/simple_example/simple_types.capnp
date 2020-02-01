@0xc4df9f8d41f26bf7;

using import "/EsiCoreAnnotations.capnp".hw;

struct EthernetFrame {
    destMAC @0 :UInt64 $hw(bits=48);
    srcMAC @1 :UInt64 $hw(bits=48);
    vlan @2 :UInt32;
    etherType @3 :UInt16;
    typePayload :union $hw( cUnion = void ) {
        ipv4 @4 :IPv4Packet;
        ipv6 @5 :IPv6Packet;
    }
}

struct IPv4Packet {

}

struct IPv6Packet {

}