using System;
using System.Collections.Generic;
using System.IO;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    public class EsiSystemVerilogCompoundWriter
    {
        protected EsiContext C;
        protected StreamWriter Writer;

        public EsiSystemVerilogCompoundWriter(EsiContext C, StreamWriter Writer)
        {
            this.C = C;
            this.Writer = Writer;
        }

        protected void W(string line = "")
        {
            Writer.WriteLine(line);
        }

        public void Write(IEnumerable<EsiCompound> compounds)
        {
            W(EsiSystemVerilogConsts.Header);
            var macro = $"__ESI_Compound_Types___";
            W($"`ifndef {macro}");
            W($"`define {macro}");
            W();

            foreach (var c in compounds)
            {
                WriteCompound(c);
                W();
            }

            W();
            W("`endif");
        }

        private void WriteCompound(EsiCompound c)
        {
            W($"typedef struct packed {{");
            if (c.Signed)
                W("  logic sign;");
            switch (c.Type)
            {
                case EsiCompound.CompoundType.EsiFixed:
                    W($"  logic [{c.Whole-1}:0] whole;");
                    W($"  logic [{c.Fractional-1}:0] frac;");
                    break;
                case EsiCompound.CompoundType.EsiFloat:
                    W($"  logic [{c.Whole-1}:0] exp;");
                    W($"  logic [{c.Fractional-1}:0] mant;");
                    break;
                default:
                    C.Log.Error("Invalid EsiCompound type: {type}", c.Type);
                    break;
            }
            W($"}} {c.GetSVCompoundModuleName()};");
        }
    }
}