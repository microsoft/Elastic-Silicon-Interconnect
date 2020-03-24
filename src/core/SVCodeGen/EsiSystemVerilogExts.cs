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

        public static string GetSVHeaderName(this EsiType type)
        {
            return type switch {
                EsiNamedType named when (!string.IsNullOrWhiteSpace(named.Name))
                    => $"{named.GetFilename()}.esi.svh",
                EsiCompound compound => "EsiCompundTypes.esi.svh",
                _ => null
            };
        }

        public static string GetSVCompoundModuleName(this EsiCompound c)
        {
            return c.Type switch {
                EsiCompound.CompoundType.EsiFixed => $"EsiFixedPoint{(c.Signed ? "S" : "U")}_w{c.Whole}_f{c.Fractional}",
                EsiCompound.CompoundType.EsiFloat => $"EsiFloatingPoint{(c.Signed ? "S" : "U")}_e{c.Whole}_m{c.Fractional}",
                _ => throw new ArgumentException($"Unhandled Compound type {Enum.GetName(typeof(EsiStruct.StructField), c.Type)}")
            };
        }

    }
}