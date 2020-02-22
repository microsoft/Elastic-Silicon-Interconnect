using System.Collections.Generic;

#nullable enable
namespace Esi.Schema
{
    public abstract class EsiType
    {
        public int? VersionOrder { get; }

        public EsiType(int? VersionOrder)
        {
            this.VersionOrder = VersionOrder;
        }
    }

    public class EsiPrimitive : EsiType
    {
        public enum PrimitiveType {
            EsiBool,
            EsiByte,
            EsiBit,
            EsiUInt,
            EsiInt
        }

        public PrimitiveType Type { get; }
        public int Bits { get; }

        public EsiPrimitive(PrimitiveType Type, int Bits, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Type = Type;
            this.Bits = Bits;
        }
    }

    public class EsiEnum : EsiType
    {
        public struct EnumMember
        {
            public string Name { get; }
            public int? Index { get; }

            public EnumMember(string Name, int? Index)
            {
                this.Name = Name;
                this.Index = Index;
            }
        }

        public IReadOnlyList<EnumMember> Members { get; }

        public EsiEnum(IReadOnlyList<EnumMember> Members, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Members = Members;
        }
    }

    public class EsiCompound : EsiType
    {
        public enum CompoundType
        {
            EsiFixed,
            EsiFloat,
            EsiUFixed,
            EsiUFloat
        }

        public CompoundType Type { get; }
        public int Whole { get; }
        public int Fractional { get; }

        public EsiCompound(CompoundType Type, int Whole, int Fractional, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Type = Type;
            this.Whole = Whole;
            this.Fractional = Fractional;
        }
    }

    public class EsiArray : EsiType
    {
        public EsiType Inner { get; }
        public int Length { get; }

        public EsiArray(EsiType Inner, int Length, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Inner = Inner;
            this.Length = Length;
        }
    }

    public class EsiStruct : EsiType
    {
        public struct StructField
        {
            public string Name { get; }
            public EsiType Type { get; }
            // BitOffset is into the _entire_ struct.
            public int? BitOffset { get; }

            public StructField(string Name, EsiType Type, int? BitOffset)
            {
                this.Name = Name;
                this.Type = Type;
                this.BitOffset = BitOffset;
            }
        }

        public string? Name { get; }
        public IReadOnlyList<StructField> Fields { get; }

        public EsiStruct(string? Name, IReadOnlyList<StructField> Fields, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Name = Name;
            this.Fields = Fields;
        }
    }

    public class EsiUnion : EsiType
    {
        public struct UnionEntry
        {
            public string Name { get; }
            public EsiType Type { get; }
        }

        public IReadOnlyList<UnionEntry> Entries { get; }
        public bool IsDiscriminated { get; }

        public EsiUnion(IReadOnlyList<UnionEntry> Entries, bool IsDiscriminated = true, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Entries = Entries;
            this.IsDiscriminated = IsDiscriminated;
        }
    }

    public class EsiList : EsiType
    {
        public EsiType Inner { get; }
        public bool IsFixed { get; }

        public EsiList(EsiType Inner, bool IsFixed = true, int? VersionOrder = null)
            : base(VersionOrder)
        {
            this.Inner = Inner;
            this.IsFixed = IsFixed;
        }
    }
}
