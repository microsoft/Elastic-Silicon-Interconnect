using System;
using Esi.Schema;

namespace Esi.CppCodeGen
{
    public static class EsiCppExts
    {
        public static string GetCppIdentifier(this EsiNamedType namedType)
        {
            return namedType?.Name
                ?.Replace('-', '_')
                ?.Replace(' ', '_');
        }

        public static string GetCppIdentifier(this EsiInterface iface)
        {
            return iface?.Name
                ?.Replace('-', '_')
                ?.Replace(' ', '_');
        }

        public static string GetCppHeaderName(this EsiNamedType type)
        {
            return $"{type.GetFilename()}.esi.hpp";
        }

        public static string GetCppInclude(this EsiType type)
        {
            return type switch {
                EsiNamedType named when (!string.IsNullOrWhiteSpace(named.Name))
                    => $"\"{named.GetCppHeaderName()}\"",
                EsiCompound compound => "\"EsiCompounds.hpp\"",
                EsiInt i => "<cstdint>",
                _ => null
            };
        }

        public static string GetCppCompoundTypeName(this EsiCompound c)
        {
            return c.Type switch {
                EsiCompound.CompoundType.EsiFloat when c.Signed == true && c.Whole == 8 && c.Fractional == 23 => "float",
                EsiCompound.CompoundType.EsiFloat when c.Signed == true && c.Whole == 11 && c.Fractional == 52 => "double",
                EsiCompound.CompoundType.EsiFloat when c.Signed == true && c.Whole == 15 && c.Fractional == 112 => "long double",
                EsiCompound.CompoundType.EsiFixed => $"EsiFixedPoint<{(c.Signed ? "true" : "false")}, {c.Whole}, {c.Fractional}>",
                EsiCompound.CompoundType.EsiFloat => $"EsiFloatingPoint<{(c.Signed ? "true" : "false")}, {c.Whole}, {c.Fractional}>",
                _ => throw new ArgumentException($"Unhandled Compound type {Enum.GetName(typeof(EsiStruct.StructField), c.Type)}")
            };
        }

    }
}