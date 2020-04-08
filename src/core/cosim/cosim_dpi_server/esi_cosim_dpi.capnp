@0xe642127a31681ef6;

interface CosimServer {
    list @0 () -> (ifaces :List(EsiInterfaceDesc));
    open @1 (iface :EsiInterfaceDesc) -> (iface :EsiInterface);
}

struct EsiInterfaceDesc {
    typeID @0 :UInt64;
    endpointID @1 :Int32;
}

interface EsiInterface {
    send @0 (blob :Data);
    recv @1 (block :Bool = true) -> (resp :Data); # If 'resp' null, no data

    close @2 ();
}