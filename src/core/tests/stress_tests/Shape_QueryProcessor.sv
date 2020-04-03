
`include "Shape.esi.svh"

module Shape_QueryProcessor (
    input logic clk,
    input logic rstn,

    IShapeQuery_getVolume_ValidReady.ParamSink param
);

    always_ff @(posedge clk)
    begin
        // do something
    end

endmodule