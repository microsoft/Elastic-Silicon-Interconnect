using System;
using System.IO;
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
            for (int i = 0; i < Indent; i++)
                Writer.Write("    ");
            Writer.WriteLine(line);
        }

        public void WriteSV(EsiType type, FileInfo headerFile)
        {
            W(EsiSystemVerilogConsts.Header);
            var macro = $"__{headerFile.Name.Replace('.', '_')}__";
            W($"`ifndef {macro}");
            W($"`define {macro}");
            W();

            WriteComment(type);
            Writer.Write("typedef ");
            WriteSVType(type);
            
            W();
            W("`endif");
        }

        private void WriteComment(EsiType type)
        {
            W("// *****");
            W($"// {type}");
            W("//");
        }

        public void WriteSVType(EsiType type)
        {
            switch (type)
            {
                case EsiStruct st:
                    WriteSVStruct(st);
                    break;
                case EsiStruct.StructField field:
                    WriteSVStructField(field);
                    break;
                default:
                    C.Log.Error("SystemVerilog interface generation for type {type} not supported", type.GetType());
                    break;
            }
        }

        private void WriteSVStructField(EsiStruct.StructField field)
        {
            W($"{field.Type.GetSVType()} {field.Name};");
        }

        public void WriteSVStruct(EsiStruct st)
        {
            W("struct packed");
            W("{");
            Indent++;
            foreach (var field in st.Fields)
            {
                WriteSVType(field);
            }
            Indent--;
            W($"}} {st.GetSVIdentifier()};");
        }
    }
}