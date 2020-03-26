using System;
using System.Collections.Generic;
using System.IO;
using Esi;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    class Program
    {
        static int Main(string[] args)
        {
            Stream input;
            bool txt = false;

            if (args.Length > 0)
            {
                input = new FileStream(args[0], FileMode.Open, FileAccess.Read);
                txt = args[0].EndsWith(".capnp");
            }
            else
            { 
                Console.WriteLine("Elastic Silicon Interconnect SystemVerilog code generator");
                Console.WriteLine("expecting binary-encoded code generation request from standard input");

                input = Console.OpenStandardInput();
            }

            using (var esiCtxt = new EsiContext())
            {
                esiCtxt.Log.Information("Starting conversion to EsiTypes");
                IEnumerable<EsiType> esiTypes;
                if (txt)
                {
                    esiTypes = EsiCapnpConvert.ConvertTextSchema(esiCtxt, new FileInfo(args[0]));
                }
                else
                {
                    esiTypes = EsiCapnpConvert.ConvertFromCGRMessage(esiCtxt, input);
                }
                esiCtxt.Log.Information("Completed reading capnp message");
                esiCtxt.Log.Information("Starting SV interface output");
                var esiSys = new EsiSystem(esiTypes);
                var sv = new EsiSystemVerilogInterfaceWriter(esiCtxt, esiSys);
                sv.WriteSVInterfaces();
                esiCtxt.Log.Information("Completed SV interface output");
            }

            return 0;
        }
    }
}
