@0xe642127a31681ef6;

interface CosimDpiServer {
    list @0 () -> (ifaces :List(EsiDpiInterfaceDesc));
    open @1 (iface :EsiDpiInterfaceDesc) -> (iface :EsiDpiEndpoint);
}

struct EsiDpiInterfaceDesc {
    typeID @0 :UInt64;
    endpointID @1 :Int32;
}

interface EsiDpiEndpoint {
    send @0 (blob :Data);
    recv @1 (block :Bool = true) -> (hasData :Bool, resp :Data); # If 'resp' null, no data

    close @2 ();
}