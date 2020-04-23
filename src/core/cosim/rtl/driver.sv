import Cosim_DpiPkg::*;

module cosim_test(
);

    localparam int TYPE_SIZE_BITS = 64;
    logic clk;
    logic rstn;

    logic DataOutValid;
    logic DataOutReady;
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

`ifndef VERILATOR

    initial
    begin
        rstn = 0;
        // No I/O activity
        DataOutReady = 0;
        DataInValid = 0;
        #17
        // Run!
        rstn = 1;
        #12

        #10
        // Accept 1 token
        DataOutReady = 1;
        #1
        @(posedge clk && DataOutValid);
        #1
        $display("Recv'd: %h", DataOut);
        DataOutReady = 0;
        #8

        #10
        // Send a token
        DataIn = 1024'hDEADBEEF;
        DataInValid = 1;
        @(posedge clk && DataInReady);
        #1
        DataInValid = 0;
        #9
        $finish();
    end

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
`endif

endmodule
