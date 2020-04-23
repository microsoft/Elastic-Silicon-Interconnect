using System.Runtime.InteropServices;
using System;
using System.Collections.Generic;
using System.IO;
using CommandLine;
using Esi;
using Esi.Schema;
using Esi.Capnp;

namespace Esi.SVCodeGen
{
    class Program
    {
        class SVCodeGenOptions : EsiCommonCommandOptions
        {
            [Option('i', "input", Required = false, HelpText = "Input Capnp file")]
            public string InputFile { get; set; } = "-";
        }

        static int Main(string[] args)
        {
            return Parser.Default.ParseArguments<SVCodeGenOptions>(args)
                    .MapResult(
                        opts => RunOpts(opts),
                        _ => 1);
        }
        private static int RunOpts(SVCodeGenOptions opts)
        {
            Stream input;
            bool txt = false;
            Directory.SetCurrentDirectory(opts.OutputDir);

            if (opts.InputFile == "-")
            { 
                Console.WriteLine("Elastic Silicon Interconnect SystemVerilog code generator");
                Console.WriteLine("expecting binary-encoded code generation request from standard input");

                input = Console.OpenStandardInput();
            }
            else
            {
                input = new FileStream(opts.InputFile, FileMode.Open, FileAccess.Read);
                txt = opts.InputFile.EndsWith(".capnp");
            }

            using (var esiCtxt = new EsiContext())
            {
                esiCtxt.Log.Information("Starting conversion to EsiTypes");
                IEnumerable<EsiObject> esiTypes;
                if (txt)
                {
                    esiTypes = EsiCapnpReader.ConvertTextSchema(esiCtxt, new FileInfo(opts.InputFile));
                }
                else
                {
                    esiTypes = EsiCapnpReader.ConvertFromCGRMessage(esiCtxt, input);
                }
                esiCtxt.Log.Information("Completed reading capnp message");
                esiCtxt.Log.Information("Starting SV interface output");
                var esiSys = new EsiSystem(esiTypes);
                var sv = new EsiSystemVerilogInterfaceWriter(esiCtxt, esiSys);
                sv.WriteSV();
                esiCtxt.Log.Information("Completed SV interface output");
            }

            return 0;
        }
    }
}
