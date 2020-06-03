#!/usr/bin/python3

import capnp
import os
import binascii
import random
import time
import subprocess
import io
import pytest

cosimDir = os.path.join(os.path.dirname(__file__), "..", "..")

class TestLoopbackBaseDPI:
    def setup_class(self):
        from test_utils import cmd
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
        self.dpi = capnp.load(os.path.join(cosimDir, "cosim_dpi_server", "esi_cosim_dpi.capnp"))
        self.rpc_client = capnp.TwoPartyClient("localhost:1111")
        self.cosim = self.rpc_client.bootstrap().cast_as(self.dpi.CosimDpiServer)
        return self.cosim

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
        ep.send(self.dpi.UntypedData.new_message(data = data)).wait()
        return data

    def read(self, ep):
        while True:
            recvResp = ep.recv(False).wait()
            if recvResp.hasData:
                break
            else:
                time.sleep(0.1)
        assert recvResp.resp is not None
        # print (recvResp)
        dataMsg = recvResp.resp.as_struct(self.dpi.UntypedData)
        # print (dataMsg)
        data = dataMsg.data
        print (binascii.hexlify(data))
        return data

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
    def test_write_read_many(self, num_msgs=50):
        ep = self.openEP()
        print ("Testing writes")
        dataSent = list()
        for _ in range(num_msgs):
            dataSent.append(self.write(ep))
        print ()
        print ("Testing reads")
        dataRecv = list()
        for _ in range(num_msgs):
            dataRecv.append(self.read(ep))
        ep.close().wait()
        assert dataSent == dataRecv

if __name__ == "__main__":
    test = TestLoopbackBaseDPI()
    test.test_write_read_many(5)
