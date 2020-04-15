import capnp
import os

cosimDir = os.path.join(os.path.dirname(__file__), "..")

def rpc():
    dpi = capnp.load(os.path.join(cosimDir, "cosim_dpi_server", "esi_cosim_dpi.capnp"))
    rpc = capnp.TwoPartyClient("localhost:1111")
    cosim = rpc.bootstrap().cast_as(dpi.CosimDpiServer)
    return (cosim, dpi)

def test_list():
    (cosim, dpi) = rpc()
    ifaces = cosim.list().wait()
    print (ifaces)
    assert len(ifaces) > 0

def test_open_close():
    (cosim, dpi) = rpc()
    ifaces = cosim.list().wait().ifaces
    print (ifaces)
    print (ifaces[0])
    openResp = cosim.open(ifaces[0]).wait()
    print (openResp)
    assert openResp.iface is not None
    ep = openResp.iface
    ep.close().wait()

def test_write():
    (cosim, dpi) = rpc()
    ifaces = cosim.list().wait().ifaces
    print (ifaces)
    print (ifaces[0])
    ep = cosim.open(ifaces[0]).wait()
    print (ep)
    assert ep.iface is not None

if __name__ == "__main__":
    test_open_close()