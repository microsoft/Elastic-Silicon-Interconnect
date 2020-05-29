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
    send @0 (msg :AnyPointer);
    recv @1 (block :Bool = true) -> (hasData :Bool, resp :AnyPointer); # If 'resp' null, no data

    close @2 ();
}

struct UntypedData {
    data @0 :Data;
}