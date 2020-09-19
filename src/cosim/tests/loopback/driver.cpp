// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "obj_dir/Vcosim_test.h"
#include "verilated_vcd_c.h"

#include <iostream>
#include <thread>

vluint64_t timeStamp;
volatile bool exitLoop = false;

double sc_time_stamp() { // Called by $time in Verilog
  return timeStamp;
}

void pollKeyboard() {
  getchar();
  exitLoop = true;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  auto &dut = *new Vcosim_test();

  Verilated::traceEverOn(true);
  VerilatedVcdC *tfp = new VerilatedVcdC;
  dut.trace(tfp, 99); // Trace 99 levels of hierarchy
  tfp->open("sim.vcd");

  auto cmdLineThread = std::thread(&pollKeyboard);

  // Reset
  dut.rstn = 0;
  dut.clk = 0;
  for (timeStamp = 1; timeStamp < 12 && !Verilated::gotFinish(); timeStamp++) {
    dut.eval();
    if (timeStamp & 1)
      dut.clk = !dut.clk;
    tfp->dump(timeStamp);
  }

  dut.rstn = 1;
  for (; !Verilated::gotFinish() && !exitLoop; timeStamp++) {
    dut.eval();
    if (timeStamp & 1)
      dut.clk = !dut.clk;
    tfp->dump(timeStamp);
    usleep(10000);
  }
  cmdLineThread.join();
  dut.final();
  // tfp->
  printf("Verilator simulation exited at time %ld\n", timeStamp);
  tfp->close();
}

void vl_stop(const char *filename, int linenum, const char *hier) VL_MT_UNSAFE {
  Verilated::gotFinish(true);
  Verilated::flushCall();
}
