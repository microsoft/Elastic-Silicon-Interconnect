# Test various ESI functionality

import pytest
from test_utils import cmd
import os

def setup_module():
    os.chdir(os.path.dirname(__file__))

class TestStress:
    @pytest.mark.nolic
    def test_svgen_verilator(self):
        cmd.run("make investData")
        cmd.run("make verilator")

    @pytest.mark.questa
    def test_svgen_questa(self):
        cmd.run("make investData")
        cmd.run("make vsim")
