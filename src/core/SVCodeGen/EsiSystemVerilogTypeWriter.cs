using System.Collections.ObjectModel;
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
        protected ISet<EsiType> Includes;
        protected IList<(EsiNamedType type, string name, string svType)> DependentAnonymousTypes;

        public EsiSystemVerilogTypeWriter(EsiContext C, StreamWriter Writer)
        {
            this.C = C;
            this.Writer = Writer;
            Includes = new HashSet<EsiType>();
            DependentAnonymousTypes = new List<(EsiNamedType type, string name, string svType)>();
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
            var svTypeString = GetStructField(
                new EsiStruct.StructField(type.GetSVIdentifier(), type),
                new List<EsiStruct.StructField>(),
                false);
            foreach (var incl in Includes
                .Select(i => i.GetSVHeaderName())
                .Where(i => !string.IsNullOrWhiteSpace(i))
                .Distinct())
            {
                W($"`include \"{incl}\"");
            }
            W();

            var macro = $"__{headerFile.Name.Replace('.', '_')}__";
            W($"`ifndef {macro}");
            W($"`define {macro}");
            W();

            if (DependentAnonymousTypes.Count() > 0)
            {
                W( "// ****");
                W($"// Types which '{type}' depends upon");
            }
            foreach (var dat in DependentAnonymousTypes)
            {
                W($"typedef {dat.svType} {dat.name};");
            }
            if (DependentAnonymousTypes.Count() > 0)
            {
                W( "// ****");
                W();
                W();
            }

            W( "// *****");
            W($"// {type}");
            W( "//");
            W($"typedef {svTypeString}");
            
            W();
            W("`endif");
            W();
            return Includes;
        }


        public string GetSVType(
            EsiType type,
            IEnumerable<EsiStruct.StructField> hierarchy,
            ref List<ulong> arrayDims,
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
                    return namedType.GetSVIdentifier();
                case EsiStruct st when useName:
                    var name = string.Join('_', hierarchy.Select(f => f.Name));
                    DependentAnonymousTypes.Add((
                        type: st,
                        name: name,
                        svType: GetSVStruct(st, hierarchy)
                    ));
                    return name;
                case EsiStruct st:
                    return GetSVStruct(st, hierarchy);

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
                    return $"logic{(i.Signed ? " signed" : " unsigned")}";

                case EsiCompound c:
                    return c.GetSVCompoundModuleName();

                case EsiArray a:
                    arrayDims.Add(a.Length);
                    return $"{GetSVType(a.Inner, hierarchy, ref arrayDims)}";

                default:
                    C.Log.Error("SystemVerilog interface generation for type {type} not supported", type.GetType());
                    return null;
            }
        }

        public string GetSVTypeSimple(
            EsiType type,
            IEnumerable<EsiStruct.StructField> hierarchy = null,
            bool useName = true)
        {
            if (hierarchy == null)
                hierarchy = new List<EsiStruct.StructField>();

            var arrayDims = new List<ulong>();
            var svString = GetSVType(type, hierarchy, ref arrayDims, useName);
            if (svString == null)
                return null;
            var sb = new StringBuilder();
            sb.Append($"{svString}");
            foreach (var d in arrayDims)
            {
                sb.Append($" [{d-1}:0]");
            }
            return sb.ToString();
        }

        protected string GetStructField(
            EsiStruct.StructField field,
            IEnumerable<EsiStruct.StructField> hierarchy,
            bool useName = true)
        {
            var newHierarchy = hierarchy.Append(field);
            var svString = GetSVTypeSimple(field.Type, newHierarchy, useName);
            if (string.IsNullOrWhiteSpace(svString))
                return $"// {field.Name} of type {field.Type}";

            return $"{svString} {field.Name};";
        }

        protected string GetSVStruct(EsiStruct st, IEnumerable<EsiStruct.StructField> hierarchy)
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