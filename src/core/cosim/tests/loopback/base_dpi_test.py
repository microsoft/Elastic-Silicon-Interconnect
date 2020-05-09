#!/usr/bin/python3

import capnp
import os
import binascii
import random
import time
import subprocess
import io
import pytest
from test_utils import cmd

cosimDir = os.path.join(os.path.dirname(__file__), "..", "..")

class TestLoopbackBaseDPI:
    def setup_class(self):
        cmd.cd_filedir(__file__)
        cmd.run("make")
        self.simPro = subprocess.Popen(["make", "run"], stdin=subprocess.PIPE)
        time.sleep(1.0)

    def teardown_class(self):
        stdin = io.TextIOWrapper(self.simPro.stdin, 'ascii')
        stdin.write("\n")
        stdin.flush()
        self.simPro.wait(1.0)

    def rpc(self):
        dpi = capnp.load(os.path.join(cosimDir, "cosim_dpi_server", "esi_cosim_dpi.capnp"))
        rpc = capnp.TwoPartyClient("localhost:1111")
        cosim = rpc.bootstrap().cast_as(dpi.CosimDpiServer)
        return cosim

    @pytest.mark.nolic
    def test_list(self):
        cosim = self.rpc()
        ifaces = cosim.list().wait().ifaces
        print (ifaces)
        assert len(ifaces) > 0

    @pytest.mark.nolic
    def test_open_close(self):
        cosim = self.rpc()
        ifaces = cosim.list().wait().ifaces
        print (ifaces)
        print (ifaces[0])
        openResp = cosim.open(ifaces[0]).wait()
        print (openResp)
        assert openResp.iface is not None
        ep = openResp.iface
        ep.close().wait()

    def write(self, ep):
        r = random.randrange(0, 2**24)
        data = r.to_bytes(3, 'big')
        print (f'Sending: {binascii.hexlify(data)}')
        ep.send(data).wait()
        return data

    def read(self, ep):
        while True:
            data = ep.recv(False).wait()
            if data.hasData:
                break
            else:
                time.sleep(0.1)
        assert data.resp is not None
        print (data)
        print (binascii.hexlify(data.resp))
        return data.resp

    def openEP(self):
        cosim = self.rpc()
        ifaces = cosim.list().wait().ifaces
        print (ifaces)
        print (ifaces[0])
        openResp = cosim.open(ifaces[0]).wait()
        print (openResp)
        assert openResp.iface is not None
        return openResp.iface

    @pytest.mark.nolic
    def test_write_read(self):
        ep = self.openEP()
        print ("Testing writes")
        dataSent = self.write(ep)
        print ()
        print ("Testing reads")
        dataRecv = self.read(ep)
        ep.close().wait()
        assert dataSent == dataRecv

    @pytest.mark.nolic
    def test_write_read_many(self):
        ep = self.openEP()
        print ("Testing writes")
        dataSent = list()
        for _ in range(50):
            dataSent.append(self.write(ep))
        print ()
        print ("Testing reads")
        dataRecv = list()
        for _ in range(50):
            dataRecv.append(self.read(ep))
        ep.close().wait()
        assert dataSent == dataRecv

if __name__ == "__main__":
    test = TestLoopbackBaseDPI()
    test.test_write_read_many()
