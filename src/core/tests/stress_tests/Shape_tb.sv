
module Shape_tb();

    logic clk;

    IShapeType_ValidReady shape();

    Shape_ComputeArea dut (
        .clk(clk),
        .rstn(1'b1),

        .shapeIn(shape.Sink)
    );

    initial begin
        #12
        shape.data.array1[1] = 39'hDEADBEEF;

        for (int i = 0; i < $size(shape.data.array1); i++) begin
            $display ("array1[%0d] = %h", i, shape.data.array1[i]);
        end
    end

    initial begin
        clk = 0;
        #10
        for (int i=0; i<10; i++) begin
            #5;
            clk = !clk;
        end
    end

endmodule