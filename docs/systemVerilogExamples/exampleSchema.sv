////////////////////
// ESI auto-generated file!
//    Do NOT edit as this file WILL be overwriten
/////////
`include "exampleSchema.svh"


// *****
// 'Polynomial3' message interface with ready/valid semantics
//
interface IPolynomial3ValidReady
    (
        input wire clk,
        input wire rstn
    );

    logic valid;
    logic ready;

    Polynomial3 data;

    modport Source (
        input clk, rstn,
        output valid,
        input ready,

        output data
    );
    
    modport Sink (
        input clk, rstn,
        input valid,
        output ready,

        input data
    );


    task send(input Polynomial3 _data, output logic success);
        data = _data;
        valid = 1;
        success = ready;
    endtask
endinterface
