using System;
using System.Collections.Generic;
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
        protected ISet<EsiNamedType> Includes;

        public EsiSystemVerilogTypeWriter(EsiContext C, StreamWriter Writer)
        {
            this.C = C;
            this.Writer = Writer;
            Indent = 0;
            Includes = new HashSet<EsiNamedType>();
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

            // Run this first to populate 'Includes'
            var svTypeString = GetSVType(type, false);
            foreach (var incl in Includes)
            {
                W($"`include \"{incl.GetSVHeaderName()}\"");
            }
            W();

            WriteComment(type);
            W($"typedef {svTypeString} {type.GetSVIdentifier()};");
            
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
                // StructField MUST go first since it is an EsiNamedType, 
                case EsiStruct.StructField field:
                    throw new ArgumentException("Internal Error: EsiStruct.StructField is not handled here!");
                case EsiNamedType namedType when (useName && !string.IsNullOrWhiteSpace(namedType.Name)):
                    Includes.Add(namedType);
                    return namedType.GetSVIdentifier();
                case EsiStruct st:
                    return GetSVStruct(st);

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
            string StructField(EsiStruct.StructField field)
            {
                var svString = GetSVType(field.Type);
                if (string.IsNullOrWhiteSpace(svString))
                    return $"// {field.Name} of type {field.Type}";
                return $"{svString} {field.Name};";
            }

            var sb = new StringBuilder();
            sb.AppendLine("struct packed {");
            Indent ++;
            foreach (var field in st.Fields)
            {
                sb.Indent(Indent).AppendLine(StructField(field));
            }
            Indent --;
            sb.Indent(Indent).Append('}');
            return sb.ToString();
        }
    }
}