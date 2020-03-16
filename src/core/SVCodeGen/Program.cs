using System;
using System.IO;
using Esi;
using Esi.Schema;

namespace SVCodeGen
{
    class Program
    {
        static int Main(string[] args)
        {
            Stream input;

            if (args.Length > 0)
            {
                input = new FileStream(args[0], FileMode.Open, FileAccess.Read);
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
                var esiTypes = EsiCapnpConvert.ConvertFromCGRMessage(esiCtxt, input);
                esiCtxt.Log.Information("Completed reading capnp message");
            }

            return 0;
        }
    }
}
