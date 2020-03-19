using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    public class EsiSystemVerilogTypeWriter
    {
        protected EsiContext C;
        protected StreamWriter Writer;
        protected int Indent;
        protected ISet<EsiType> Includes;

        public EsiSystemVerilogTypeWriter(EsiContext C, StreamWriter Writer)
        {
            this.C = C;
            this.Writer = Writer;
            Indent = 0;
            Includes = new HashSet<EsiType>();
        }

        protected void W(string line = "")
        {
            Writer.WriteLine(line);
        }

        /// <summary>
        /// Write out a SV header with a typedef for the type
        /// </summary>
        /// <param name="type"></param>
        /// <param name="headerFile"></param>
        /// <returns>A set of the used types</returns>
        public ISet<EsiType> WriteSV(EsiNamedType type, FileInfo headerFile)
        {
            W(EsiSystemVerilogConsts.Header);
            var macro = $"__{headerFile.Name.Replace('.', '_')}__";
            W($"`ifndef {macro}");
            W($"`define {macro}");
            W();

            // Run this first to populate 'Includes'
            var svTypeString = GetStructField(
                new EsiStruct.StructField(type.GetSVIdentifier(), type), false);
            foreach (var incl in Includes
                .Select(i => i.GetSVHeaderName())
                .Where(i => !string.IsNullOrWhiteSpace(i))
                .Distinct())
            {
                W($"`include \"{incl}\"");
            }
            W();

            WriteComment(type);
            W($"typedef {svTypeString}");
            
            W();
            W("`endif");

            return Includes;
        }

        private void WriteComment(EsiType type)
        {
            W( "// *****");
            W($"// {type}");
            W( "//");
        }

        public string GetSVType(EsiType type, ref List<ulong> arrayDims, bool useName = true)
        {
            if (useName)
                Includes.Add(type);
            switch (type)
            {
                // StructField MUST go first since it is an EsiNamedType, 
                case EsiStruct.StructField field:
                    throw new ArgumentException("Internal Error: EsiStruct.StructField is not handled here!");
                // Refer to another named struct when 1) it actually has a name and 2) we're permitted to
                case EsiNamedType namedType when (useName && !string.IsNullOrWhiteSpace(namedType.Name)):
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
                    arrayDims.Add(i.Bits);
                    return $"logic";

                case EsiCompound c:
                    return c.GetSVCompoundModuleName();

                case EsiArray a:
                    arrayDims.Add(a.Length);
                    return $"{GetSVType(a.Inner, ref arrayDims)}";

                default:
                    C.Log.Error("SystemVerilog interface generation for type {type} not supported", type.GetType());
                    return null;
            }
        }

        protected string GetStructField(EsiStruct.StructField field, bool useName = true)
        {
            var arrayDims = new List<ulong>();
            var svString = GetSVType(field.Type, ref arrayDims, useName);
            if (string.IsNullOrWhiteSpace(svString))
                return $"// {field.Name} of type {field.Type}";

            var sb = new StringBuilder();
            sb.Append($"{svString}");
            foreach (var d in arrayDims)
            {
                sb.Append($"[{d-1}:0]");
            }
            sb.Append($" {field.Name};");
            return sb.ToString();
        }

        protected string GetSVStruct(EsiStruct st)
        {
            var sb = new StringBuilder();
            sb.AppendLine("struct packed {");
            Indent ++;
            foreach (var field in st.Fields)
            {
                sb.Indent(Indent).AppendLine(GetStructField(field));
            }
            Indent --;
            sb.Indent(Indent).Append('}');
            return sb.ToString();
        }
    }
}