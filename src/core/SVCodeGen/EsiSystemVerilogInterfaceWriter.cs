using System;
using System.Collections.Generic;
using System.IO;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    public class EsiSystemVerilogInterfaceWriter
    {
        protected EsiContext C { get; }
        protected EsiSystem Sys { get; }
        public EsiSystemVerilogInterfaceWriter(EsiContext ctxt, EsiSystem sys)
        {
            C = ctxt;
            Sys = sys;
        }

        public void WriteSVInterfaces(DirectoryInfo to = null)
        {
            to = to ?? new DirectoryInfo(Directory.GetCurrentDirectory());

            foreach (var type in Sys.NamedTypes.Values)
            {
                var headerFile = to.FileUnder($"{type.GetFilename()}.esi.svh");
                WriteSVType(type, headerFile);
                WriteSVInterface(type, to.FileUnder($"{type.GetFilename()}.esi.sv"), headerFile);
            }
        }

        public void WriteSVType(EsiNamedType type, FileInfo fileInfo)
        {
            using (var write = new StreamWriter(fileInfo.OpenWrite()))
            {
                var svTypeWriter = new EsiSystemVerilogTypeWriter(C, write);
                svTypeWriter.WriteSV(type, fileInfo);
            }
        }

        public void WriteSVInterface(EsiType type, FileInfo fileInfo, FileInfo headerFile)
        {
            using (var write = new StreamWriter(fileInfo.OpenWrite()))
            {
                write.WriteLine(EsiSystemVerilogConsts.Header);
                write.Write($@"
`include ""{headerFile.Name}""

interface I{type.GetSVType()}ValidReady
    (
        input wire clk,
        input wire rstn
    );

    logic valid;
    logic ready;

    {type.GetSVType()} data;

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

endinterface
                ");
            }
        }
    }
}