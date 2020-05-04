#include "obj_dir/Vcosim_test.h"
#include "verilated_vcd_c.h"

#include <iostream>

vluint64_t timeStamp;

double sc_time_stamp () {       // Called by $time in Verilog
    return timeStamp;
}

int main(int argc, char** argv)
{
    Verilated::commandArgs(argc, argv);
    auto& dut = *new Vcosim_test();

    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    dut.trace(tfp, 99);  // Trace 99 levels of hierarchy
    tfp->open("sim.vcd");

    // Reset
    dut.rstn = 0;
    dut.clk = 0;
    for (timeStamp = 1; timeStamp<11 && !Verilated::gotFinish(); timeStamp++)
    {
        dut.eval();
        dut.clk = !dut.clk;
        tfp->dump(timeStamp);
    }

    dut.rstn = 1;
    for (; !Verilated::gotFinish(); timeStamp++)
    {
        dut.eval();
        dut.clk = !dut.clk;
        tfp->dump(timeStamp);
        usleep(10000);
    }
    dut.final();
    // tfp->
    printf("Verilator simulation exited at time %ld\n", timeStamp);
    tfp->close();
}

void vl_stop(const char* filename, int linenum, const char* hier) VL_MT_UNSAFE {
    Verilated::gotFinish(true);
    Verilated::flushCall();
}
