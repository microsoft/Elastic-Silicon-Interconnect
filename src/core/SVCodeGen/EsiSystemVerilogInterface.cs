using System;
using System.Collections.Generic;
using System.IO;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    public class EsiSystemVerilogInterface
    {
        protected EsiContext C { get; }
        protected EsiSystem Sys { get; }
        public EsiSystemVerilogInterface(EsiContext ctxt, EsiSystem sys)
        {
            C = ctxt;
            Sys = sys;
        }

        public void WriteSVInterfaces(DirectoryInfo to = null)
        {
            to = to ?? new DirectoryInfo(Directory.GetCurrentDirectory());

            foreach (var type in Sys.NamedTypes.Values)
            {
                WriteSVInterface(type, to.FileUnder($"{type.Name}.esi.sv"));
            }
        }

        public void WriteSVInterface(EsiNamedType type, FileInfo fileInfo)
        {
            using (var write = fileInfo.OpenWrite())
            {
                switch (type)
                {
                    case EsiStruct st:
                        WriteSVInterface(st, write);
                        break;
                    default:
                        C.Log.Error("SystemVerilog interface generation for top level type {type} not supported", type.GetType());
                        break;
                }
            }
        }

        public void WriteSVInterface(EsiStruct st, Stream write)
        {
            throw new NotImplementedException();
        }
    }
}