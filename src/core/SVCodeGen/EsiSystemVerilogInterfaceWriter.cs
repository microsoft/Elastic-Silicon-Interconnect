using System.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using Esi.Schema;
using RazorLight;


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

        public void WriteSV(DirectoryInfo to = null)
        {
            var usedTypes = new HashSet<EsiType>();
            to = to ?? new DirectoryInfo(Directory.GetCurrentDirectory());

            // Output the SV interfaces and structs for all the types
            foreach (var type in Sys.NamedTypes.Values)
            {
                var headerFile = to.FileUnder(type.GetSVHeaderName());
                var usedTypesLocal = WriteSVType(type, headerFile);
                usedTypes.UnionWith(usedTypesLocal);

                WriteSVTypeInterface(type, to.FileUnder($"{type.GetFilename()}.esi.sv"), headerFile);
            }

            // During the type output process, various shared types were used
            // which don't exist yet. Put them all in the same file here.
            var usedCompounds = usedTypes.Where(t => t is EsiCompound).Select(t => t as EsiCompound).Distinct();
            if (usedCompounds.Count() > 0)
            {
                C.Log.Information("Writing compound types to single file");
                var compoundTypesFile = to.FileUnder(usedCompounds.First().GetSVHeaderName());
                if (compoundTypesFile.Exists)
                    compoundTypesFile.Delete();
                using (var writer = new StreamWriter(compoundTypesFile.OpenWrite()))
                {
                    var compoundWriter = new EsiSystemVerilogCompoundWriter(C, writer);
                    compoundWriter.Write(usedCompounds);
                }
            }

            // For each interface, write a system verilog interface
            foreach (var iface in Sys.Interfaces)
            {
                WriteSVInterface(iface, to.FileUnder($"Interface{iface.Name}.esi.sv"));
            }
        }

        public ISet<EsiType> WriteSVType(EsiNamedType type, FileInfo fileInfo)
        {
            if (fileInfo.Exists)
                fileInfo.Delete();
            using (var write = new StreamWriter(fileInfo.OpenWrite()))
            {
                C.Log.Information("Starting SV type generation for {type} to file {file}",
                    type, fileInfo.Name);
                var svTypeWriter = new EsiSystemVerilogTypeWriter(C, write);
                return svTypeWriter.WriteSV(type, fileInfo);
            }
        }


        public void WriteSVTypeInterface(EsiNamedType type, FileInfo fileInfo, FileInfo headerFile)
        {
            C.Log.Information("Starting SV interface generation for {type} to file {file}",
                type, fileInfo.Name);
            var model = (Type: type, HeaderFile: headerFile);
            RazorEngine.Engine.RenderToFile("sv/type_interface.sv", model, fileInfo);
        }

        public void WriteSVInterface(EsiInterface iface, FileInfo to)
        {
            if (to.Exists)
                to.Delete();

            using (var write = new StreamWriter(to.OpenWrite()))
            {
                var svTypeWriter = new EsiSystemVerilogTypeWriter(C, write);
                C.Log.Information("Starting SV interface generation for {iface} to file {file}",
                    iface, to.Name);
                write.WriteLine(EsiSystemVerilogConsts.Header);

                write.WriteLine();

                var usedTypes = new List<EsiType>();
                usedTypes.AddRange(iface.Methods.SelectMany(m => m.Params.Select(p => p.Type)));
                usedTypes.AddRange(iface.Methods.SelectMany(m => m.Returns.Select(p => p.Type)));
                foreach (var usedNamedType in usedTypes.Distinct())
                {
                    var header = usedNamedType.GetSVHeaderName();
                    if (!string.IsNullOrWhiteSpace(header))
                        write.WriteLine($"`include \"{header}\"");
                }
                write.WriteLine();


                foreach (var method in iface.Methods)
                {
                    write.WriteLine();
                    write.WriteLine($"///");
                    write.WriteLine($"/// Interface '{iface.Name}' method '{method.Name}'");
                    write.WriteLine($"///");
                    write.WriteLine($"interface I{iface.GetSVIdentifier()}_{method.Name}_ValidReady ();");
                    write.WriteLine();
                    if (method.Params?.Count() > 0)
                    {
                        write.WriteLine("    // Input parameters (all signals are prefixed with 'p')");
                        WriteParamReturn(method.Params, true);
                    }
                    if (method.Returns?.Count() > 0)
                    {
                        write.WriteLine();
                        write.WriteLine("    // Output returns (all signals are prefixed with 'r')");
                        WriteParamReturn(method.Returns, false);
                    }

                    write.WriteLine("endinterface");
                }

                void WriteParamReturn((string Name, EsiType Type)[] pr, bool isParam)
                {
                    var pvChar = isParam ? "p" : "r";
                    var pvString = isParam ? "Param" : "Return";
                    write.WriteLine($"    logic {pvChar}Valid;");
                    write.WriteLine($"    logic {pvChar}Ready;");
                    foreach (var p in pr)
                        write.WriteLine($"    {svTypeWriter.GetSVTypeSimple(p.Type, useName: true)} {pvChar}{p.Name};");

                    write.WriteLine();
                    write.WriteLine($"    modport {pvString}Source (");
                    foreach (var p in pr)
                        write.WriteLine($"        output {pvChar}{p.Name},");

                    write.WriteLine($"        output {pvChar}Valid,");
                    write.WriteLine($"        input {pvChar}Ready");
                    write.WriteLine( "    );");

                    write.WriteLine($"    modport {pvString}Sink (");
                    foreach (var p in pr)
                        write.WriteLine($"        input {pvChar}{p.Name},");

                    write.WriteLine($"        input {pvChar}Valid,");
                    write.WriteLine($"        output {pvChar}Ready");
                    write.WriteLine( "    );");
                    write.WriteLine();
                }

            }
        }
    }
}