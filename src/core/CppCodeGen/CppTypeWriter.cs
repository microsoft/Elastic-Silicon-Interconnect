using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Esi.Schema;

namespace Esi.CppCodeGen
{
    public class EsiCppTypeWriter
    {
        protected EsiContext C;
        protected FileInfo HeaderFile;
        protected StreamWriter Writer;
        protected ISet<EsiType> Includes;

        public EsiCppTypeWriter(EsiContext C, FileInfo HeaderFile, StreamWriter Writer)
        {
            this.C = C;
            this.Writer = Writer;
            this.HeaderFile = HeaderFile;
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
        public void WriteCppHeader(EsiNamedType type)
        {
            W(EsiCppConsts.Header);

            var macro = $"__{HeaderFile.Name.Replace('.', '_')}__";
            W($"#ifndef {macro}");
            W($"#define {macro}");
            W();

            W();
            W("// ---");
            W("// Type description plain text");
            W("//");
            var textDescriptionBuilder = new StringBuilder();
            type.GetDescriptionTree(textDescriptionBuilder, 0);
            foreach (var line in textDescriptionBuilder
                .ToString()
                .Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                W($"// {line.TrimEnd()}");
            }
            W();

            // Run this first to populate 'Includes'
            var cppTypeString = GetStructField(
                new EsiStruct.StructField(type.GetCppIdentifier(), type),
                new List<EsiStruct.StructField>(),
                false);
            foreach (var incl in Includes
                .Select(i => i.GetCppInclude())
                .Where(i => !string.IsNullOrWhiteSpace(i))
                .Distinct())
            {
                W($"#include {incl}");
            }
            W();

            W( "// *****");
            W($"// {type}");
            W( "//");
            W($"typedef {cppTypeString}");

            W();
            W("#endif");
        }

        public string GetCppType(
            EsiType type,
            IEnumerable<EsiStruct.StructField> hierarchy,
            bool useName = true)
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
                    return namedType.GetCppIdentifier();
                case EsiStruct st:
                    return GetCppStruct(st, hierarchy);

                // -----
                // Simple types
                case EsiPrimitive p when (p.Type == EsiPrimitive.PrimitiveType.EsiByte):
                    return "unsigned char";
                case EsiPrimitive p when (p.Type == EsiPrimitive.PrimitiveType.EsiVoid):
                    return null;
                case EsiPrimitive p:
                    return "bool";
                case EsiInt i when i.Bits <= 8:
                    Includes.Add(i);
                    return $"{(i.Signed ? "" : "u")}int8_t";
                case EsiInt i when i.Bits <= 16:
                    Includes.Add(i);
                    return $"{(i.Signed ? "" : "u")}int16_t";
                case EsiInt i when i.Bits <= 32:
                    Includes.Add(i);
                    return $"{(i.Signed ? "" : "u")}int32_t";
                case EsiInt i when i.Bits <= 64:
                    Includes.Add(i);
                    return $"{(i.Signed ? "" : "u")}int64_t";

                case EsiCompound c:
                    return c.GetCppCompoundTypeName();

                case EsiArray a:
                    return $"{GetCppType(a.Inner, hierarchy)}[{a.Length}]";

                default:
                    C.Log.Error("C++ type generation for type {type} not supported", type.GetType());
                    return null;
            }
        }

        public string GetCppTypeSimple(
            EsiType type,
            IEnumerable<EsiStruct.StructField> hierarchy = null,
            bool useName = true)
        {
            if (hierarchy == null)
                hierarchy = new List<EsiStruct.StructField>();

            var arrayDims = new List<ulong>();
            return GetCppType(type, hierarchy, useName);
        }

        protected string GetStructField(
            EsiStruct.StructField field,
            IEnumerable<EsiStruct.StructField> hierarchy,
            bool useName = true)
        {
            var newHierarchy = hierarchy.Append(field);
            var svString = GetCppTypeSimple(field.Type, newHierarchy, useName);
            if (string.IsNullOrWhiteSpace(svString))
                return $"// {field.Name} of type {field.Type}";

            return $"{svString} {field.Name};";
        }

        protected string GetCppStruct(EsiStruct st, IEnumerable<EsiStruct.StructField> hierarchy)
        {
            var sb = new StringBuilder();
            sb.AppendLine("struct packed {");
            foreach (var field in st.Fields.Reverse())
            {
                sb.Indent(2).AppendLine(GetStructField(field, hierarchy));
            }
            sb.Append('}');
            return sb.ToString();
        }

    }
}