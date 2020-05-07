# Test various ESI functionality

import pytest
from test_utils import cmd
import os
import shutil

workDir = os.path.join(os.path.dirname(__file__), "test_tmp")
def setup_module():
    cmd.cd_filedir(__file__)

class TestStress:
    @pytest.mark.nolic
    def test_svgen_verilator(self):
        cmd.run("make investData")
        cmd.run("make verilator")

    @pytest.mark.questa
    def test_svgen_questa(self):
        cmd.run("make investData")
        cmd.run("make vsim")

    @pytest.mark.hwlib
    def test_shape_questa(self):
        cmd.run("hwbuild -d %s -n shape.hwlib.yml questa" % workDir)

    @pytest.mark.hwlib
    def test_shape_quartus(self):
        cmd.run("hwbuild -d %s -n shape.hwlib.yml -r intel.Arria10 quartus -c map" % workDir)
