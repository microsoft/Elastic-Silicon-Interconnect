
`include "Shape.esi.svh"

module Shape_ComputeArea (
    input logic clk,
    input logic rstn,

    IShapeType_ValidReady.Sink shapeIn
);

    always_ff @(posedge clk)
    begin
        // do something
    end

endmodule