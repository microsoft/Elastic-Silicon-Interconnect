using System;
using System.IO;
using Esi.Schema;

namespace Esi.CppCodeGen
{
    public class CppApiWriter
    {
        protected EsiContext C;
        protected EsiSystem Sys;
        public CppApiWriter(EsiContext ctxt, EsiSystem Sys)
        {
            this.Sys = Sys;
            this.C = ctxt;
        }

        public void WriteAPI(DirectoryInfo to = null)
        {
            to = to ?? new DirectoryInfo(Directory.GetCurrentDirectory());
            var compoundTypesHeader = new FileInfo(Path.Join(
                Esi.Utils.RootDir.FullName,
                "CppCodeGen", "support_includes", "EsiCompounds.hpp"));
            compoundTypesHeader.CopyTo(Path.Join(to.FullName, compoundTypesHeader.Name), true);
            foreach (var type in Sys.NamedTypes.Values)
            {
                WriteType(type, to);
            }
        }

        public void WriteType(EsiNamedType type, DirectoryInfo to)
        {
            var typeName = type.Name;
            var headerFile = to.FileUnder(type.GetCppHeaderName());
            if (headerFile.Exists)
                headerFile.Delete();
            using (var writer = new StreamWriter(headerFile.OpenWrite()))
            {
                var typeWriter = new EsiCppTypeWriter(C, headerFile, writer);
                typeWriter.WriteCppHeader(type);
            }
        }
    }
}