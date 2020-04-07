import CosimCore_DpiPkg::*;

module cosim_test(
);

    logic clk;
    Cosim_Endpoint #(
        .ENDPOINT_ID(1),
        .ESI_TYPE_ID(1),
        .TYPE_SIZE_BITS(1026)
    ) ep (
        .clk(clk),
        .rstn(1'b1),

        .DataOutReady(1'b1)
    );

`ifndef VERILATOR
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
