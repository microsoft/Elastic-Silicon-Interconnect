
// ******
// This file is hand-coded
`include "exampleSchema.svh"

// Compute ax^2 + bx + c = y
module Polynomial3Compute (
    input logic clk,
    input logic rstn,

    // ESI connection
    IPolynomial3ValidReady.Sink abc,
    // input Polynomial3 abc,

    // Legacy wires
    input logic[9:0] x,
    output logic[45:0] y
);

    Polynomial3 dr;
    logic[9:0] xr;
    assign abc.ready = 1'b1;

    always_ff @(posedge clk) begin
        if (abc.valid) begin
            dr <= abc.data;
            xr <= x;
        end
    end

    always_comb begin
        y = (dr.a*xr*xr) + (dr.b*xr) + dr.c;
    end

endmodule 



///
/// Verilator cannot handle a top-level module which has a modport.
/// This wrapper will be necessary for simulations.
///
module Polynomial3Compute_WireWrapper (
    input logic clk,
    input logic rstn,

    input Polynomial3 abcData,
    input logic[9:0] x,
    output logic[45:0] y
);

    IPolynomial3ValidReady inputAbc (.clk(clk), .rstn(1'b1));
    assign inputAbc.data = abcData;

    Polynomial3Compute dut (
        .clk(clk),
        .rstn(1'b1),

        .abc(inputAbc.Source),
        .x(x),

        .y(y)
    );

endmodule
