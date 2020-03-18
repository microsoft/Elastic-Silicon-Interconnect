using System;
using System.IO;
using System.Text;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    public class EsiSystemVerilogTypeWriter
    {
        protected EsiContext C;
        protected StreamWriter Writer;
        protected int Indent;

        public EsiSystemVerilogTypeWriter(EsiContext C, StreamWriter Writer)
        {
            this.C = C;
            this.Writer = Writer;
            Indent = 0;
        }

        protected void W(string line = "")
        {
            Writer.WriteLine(line);
        }

        public void WriteSV(EsiNamedType type, FileInfo headerFile)
        {
            W(EsiSystemVerilogConsts.Header);
            var macro = $"__{headerFile.Name.Replace('.', '_')}__";
            W($"`ifndef {macro}");
            W($"`define {macro}");
            W();

            WriteComment(type);
            W($"typedef {GetSVType(type, false)}");
            W($"{type.GetSVIdentifier()};");
            
            W();
            W("`endif");
        }

        private void WriteComment(EsiType type)
        {
            W( "// *****");
            W($"// {type}");
            W( "//");
        }

        public string GetSVType(EsiType type, bool useName = true)
        {
            switch (type)
            {
                case EsiNamedType namedType when (useName && !string.IsNullOrWhiteSpace(namedType.Name)):
                    return namedType.GetSVIdentifier();
                case EsiStruct st:
                    return GetSVStruct(st);
                case EsiStruct.StructField field:
                    return $"{GetSVType(field.Type)} {field.Name};";


                // -----
                // Simple types
                case EsiPrimitive p when (p.Type == EsiPrimitive.PrimitiveType.EsiByte):
                    return "logic [7:0]";
                case EsiPrimitive p when (p.Type == EsiPrimitive.PrimitiveType.EsiVoid):
                    return null;
                case EsiPrimitive p:
                    return "logic";
                case EsiInt i:
                    return $"logic [{i.Bits-1}:0]";


                default:
                    C.Log.Error("SystemVerilog interface generation for type {type} not supported", type.GetType());
                    return null;
            }
        }

        protected string GetSVStruct(EsiStruct st)
        {
            var sb = new StringBuilder();
            sb.AppendLine("struct packed {");
            Indent ++;
            foreach (var field in st.Fields)
            {
                sb.Indent(Indent).AppendLine(GetSVType(field, false));
            }
            Indent --;
            sb.Indent(Indent).Append('}');
            return sb.ToString();
        }
    }
}