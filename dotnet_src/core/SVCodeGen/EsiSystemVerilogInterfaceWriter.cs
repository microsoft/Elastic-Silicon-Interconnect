// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using Esi.Schema;
using Scriban.Runtime;

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
            C.Log.Information("Starting SV type generation for {type} to file {file}",
                type, fileInfo.Name);
            var svTypeWriter = new EsiSystemVerilogTypeWriter(C);
            return svTypeWriter.WriteSVHeader(type, fileInfo);
        }

        public void WriteSVTypeInterface(EsiNamedType type, FileInfo to, FileInfo headerFile)
        {
            var s = new ScriptObject();
            s.Add("header", headerFile);
            s.Add("type", type);
            SVUtils.RenderTemplate("sv/type_interface.sv.sbntxt", s, to);
        }

        public void WriteSVInterface(EsiInterface iface, FileInfo to)
        {
            var svTypeWriter = new EsiSystemVerilogTypeWriter(C);
            string SimpleTypeString(EsiType type)
            {
                try {
                    return svTypeWriter.GetSVTypeSimple(type, useName: true);
                } catch (Exception e)
                {
                    C.Log.Error("Exception in getting a typestring for '{type}': {e}",
                        type, e);
                    return "<exception>";
                }
            }

            C.Log.Information("Starting SV interface generation for {iface} to file {file}",
                iface, to.Name);

            var paramTypes = iface.Methods.SelectMany(m => m.Params.Select(p => p.Type));
            var returnTypes = iface.Methods.SelectMany(m => m.Returns.Select(p => p.Type));
            var usedTypes = paramTypes.Concat(returnTypes).Distinct();

            var s = new ScriptObject();
            s.Add("iface", iface);
            s.Add("usedTypes", usedTypes);
            s.Import("SimpleTypeString", new Func<EsiType, string>(SimpleTypeString));

            SVUtils.RenderTemplate("sv/full_interface.sv.sbntxt", s, to);
        }
    }
}