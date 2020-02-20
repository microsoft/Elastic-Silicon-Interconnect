module Polynomial3Compute_tb ();

    logic clk;

    IPolynomial3ValidReady inputAbc (.clk(clk), .rstn(1'b1));
    logic [9:0] inputX;
    logic [45:0] outputY;

    Polynomial3Compute dut (
        .clk(clk),
        .rstn(1'b1),

        .abc(inputAbc.Sink),
        .x(inputX),

        .y(outputY)
    );

    initial begin
        clk = 0;
        #10
        for (int i=0; i<10; i++) begin
            #5;
            clk = !clk;
        end
    end

    initial begin
        #17
        inputAbc.valid = 1;
        inputAbc.data.a = 42;
        inputAbc.data.b = 184;
        inputAbc.data.c = 2;
        inputX = 1;
    end

endmodule