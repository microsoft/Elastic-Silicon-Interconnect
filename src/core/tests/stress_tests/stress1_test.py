# Test various ESI functionality

from test_utils import cmd
import os

def setup_module():
    os.chdir(os.path.dirname(__file__))

class TestStress:

    def test_svgen(self):
        cmd.run("make investData")
        cmd.run("make run")
        cmd.run("make verilator")
