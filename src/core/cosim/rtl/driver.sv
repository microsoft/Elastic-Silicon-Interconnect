import Cosim_DpiPkg::*;

module cosim_test(
    `ifdef VERILATOR
    input logic clk,
    input logic rstn
    `endif
);
    // initial begin
    //     if ($test$plusargs("trace") != 0) begin
    //         $display("[%0t] Tracing to sim.vcd...\n", $time);
    //         $dumpfile("sim.vcd");
    //         $dumpvars();
    //     end
    // end

    localparam int TYPE_SIZE_BITS = 23;
    `ifndef VERILATOR
    logic clk;
    logic rstn;
    `endif

    logic DataOutValid;
    logic DataOutReady = 1;
    logic[TYPE_SIZE_BITS-1:0] DataOut;
    
    logic DataInValid;
    logic DataInReady;
    logic [TYPE_SIZE_BITS-1:0] DataIn;

    Cosim_Endpoint #(
        .ENDPOINT_ID(1),
        .ESI_TYPE_ID(1),
        .TYPE_SIZE_BITS(TYPE_SIZE_BITS)
    ) ep (
        .*
    );

    always@(posedge clk)
    begin
        if (rstn)
        begin
            if (DataOutValid && DataOutReady)
            begin
                $display("Recv'd: %h", DataOut);
                DataIn <= DataOut;
                DataInValid <= 1;
            end

            if (DataInValid && DataInReady)
            begin
                $display("Sent: %h", DataIn);
                DataInValid <= 0;
                DataIn <= 'x;
            end
        end
        else
        begin
            DataInValid <= 0;
        end
    end

    //     #10
    //     // Accept 1 token
    //     DataOutReady = 1;
    //     #1
    //     @(posedge clk && DataOutValid);
    //     #1
    //     $display("Recv'd: %h", DataOut);
    //     DataOutReady = 0;
    //     #8

    //     #10
    //     // Send a token
    //     DataIn = 1024'hDEADBEEF;
    //     DataInValid = 1;
    //     @(posedge clk && DataInReady);
    //     #1
    //     DataInValid = 0;
    //     #9
    //     $finish();
    // end

`ifndef VERILATOR
    // Clock
    initial
    begin
        clk = 1'b0;
        while (1)
        begin
            #5;
            clk = !clk;
        end
    end

    initial
    begin
        rstn = 0;
        #17
        // Run!
        rstn = 1;
    end
`endif

endmodule
