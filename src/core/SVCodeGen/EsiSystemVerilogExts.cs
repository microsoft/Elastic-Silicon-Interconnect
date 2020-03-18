using System;
using System.Text;
using Esi.Schema;

namespace Esi.SVCodeGen
{
    public static class EsiSystemVerilogExts
    {
        public static string GetSVIdentifier(this EsiNamedType namedType)
        {
            return namedType?.Name
                ?.Replace('-', '_')
                ?.Replace(' ', '_');
        }

        public static StringBuilder Indent(this StringBuilder stringBuilder, int indent)
        {
            for (int i = 0; i < indent; i++)
                stringBuilder.Append("  ");
            return stringBuilder;
        }
    }
}