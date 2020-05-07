#!/usr/bin/python3

import capnp
import os
import binascii
import random

cosimDir = os.path.join(os.path.dirname(__file__), "..")

def rpc():
    dpi = capnp.load(os.path.join(cosimDir, "cosim_dpi_server", "esi_cosim_dpi.capnp"))
    rpc = capnp.TwoPartyClient("localhost:1111")
    cosim = rpc.bootstrap().cast_as(dpi.CosimDpiServer)
    return (cosim, dpi)

def test_list():
    (cosim, dpi) = rpc()
    ifaces = cosim.list().wait().ifaces
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
    openResp = cosim.open(ifaces[0]).wait()
    print (openResp)
    assert openResp.iface is not None
    ep = openResp.iface
    # data = bytes.fromhex('D9 5E FF')
    r = random.randrange(0, 2**24)
    data = r.to_bytes(3, 'big')
    print (f'Sending: {binascii.hexlify(data)}')
    ep.send(data).wait()
    ep.close().wait()

def test_read():
    (cosim, dpi) = rpc()
    ifaces = cosim.list().wait().ifaces
    print (ifaces)
    print (ifaces[0])
    openResp = cosim.open(ifaces[0]).wait()
    print (openResp)
    assert openResp.iface is not None
    ep = openResp.iface
    data = ep.recv(False).wait().resp
    print (binascii.hexlify(data))
    assert data is not None
    ep.close().wait()

if __name__ == "__main__":
    print ("Testing writes")
    test_write()

    print ()
    print ("Testing reads")
    test_read()
