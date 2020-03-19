
`include "Shape.esi.svh"

module Shape_ComputeArea (
    input logic clk,
    input logic rstn,

    IShapeValidReady.Sink shapeIn
);

    always_ff @(posedge clk)
    begin
        // do something
    end

endmodule