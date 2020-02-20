using System;
using System.IO;

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

            return 0;
        }
    }
}
