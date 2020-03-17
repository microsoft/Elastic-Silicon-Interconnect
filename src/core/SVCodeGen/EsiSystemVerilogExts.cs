using System;
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

        public static string GetSVType(this EsiType type)
        {
            switch (type)
            {
                case EsiNamedType namedType:
                    return GetSVIdentifier(namedType);
                case EsiPrimitive p when (p.Type == EsiPrimitive.PrimitiveType.EsiByte):
                    return "logic [7:0]";
                case EsiPrimitive p when (p.Type == EsiPrimitive.PrimitiveType.EsiVoid):
                    return null;
                case EsiPrimitive p:
                    return "logic";
                case EsiInt i:
                    return $"logic [{i.Bits-1}:0]";
                default:
                    throw new NotImplementedException($"Type {type.GetType()} is not implemented for SystemVerilog");
            }
        }
    }
}