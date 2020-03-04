using System.Linq;
using System;
using System.Collections.Generic;

#nullable enable
namespace Esi.Schema
{
    public abstract class EsiType
    {    }

    public class EsiPrimitive : EsiType
    {
        public enum PrimitiveType {
            EsiVoid,
            EsiBool,
            EsiByte,
            EsiBit,
        }

        public PrimitiveType Type { get; }

        public EsiPrimitive(PrimitiveType Type)
            : base()
        {
            this.Type = Type;
        }
    }

    public class EsiInt : EsiType
    {
        public bool Signed { get; }
        public ulong Bits { get; }

        public EsiInt(ulong Bits, bool Signed)
            : base()
        {
            this.Bits = Bits;
            this.Signed = Signed;
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

        public EsiEnum(IReadOnlyList<EnumMember> Members)
            : base()
        {
            this.Members = Members;
        }
    }

    public class EsiCompound : EsiType
    {
        public enum CompoundType
        {
            EsiFixed,
            EsiFloat
        }

        public CompoundType Type { get; }
        public bool Signed { get; }
        public int Whole { get; }
        public int Fractional { get; }

        public EsiCompound(CompoundType Type, bool Signed, int Whole, int Fractional)
            : base()
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

        public EsiArray(EsiType Inner, int Length)
            : base()
        {
            this.Inner = Inner;
            this.Length = Length;
        }

        public EsiArray(Func<EsiType> Inner, int Length)
            : base()
        {
            this.Inner = Inner();
            this.Length = Length;
        }
    }

    public class EsiStruct : EsiType
    {
        public struct StructField
        {
            public string Name { get; }
            // This is used for versioning in CapnProto
            public ulong? CodeOrder { get; }
            public EsiType Type { get; }
            // BitOffset is into the _entire_ struct.
            public int? BitOffset { get; }

            public StructField(string Name, EsiType Type, ulong? CodeOrder = null, int? BitOffset = null)
            {
                this.Name = Name;
                this.Type = Type;
                this.CodeOrder = CodeOrder;
                this.BitOffset = BitOffset;
            }

            public StructField(string Name, Func<EsiType> Type, ulong? CodeOrder = null, int? BitOffset = null)
            {
                this.Name = Name;
                this.Type = Type();
                this.CodeOrder = CodeOrder;
                this.BitOffset = BitOffset;
            }
        }

        public string? Name { get; }
        public StructField[] Fields { get; }

        public EsiStruct(string? Name, IEnumerable<StructField> Fields)
            : base()
        {
            this.Name = Name;
            this.Fields = Fields.ToArray();
        }

        public EsiStruct(string? Name, Func<EsiStruct, IEnumerable<StructField>> Fields)
            : base()
        {
            this.Name = Name;
            this.Fields = Fields(this).ToArray();
        }
    }

    public class EsiUnion : EsiType
    {
        public struct UnionEntry
        {
            public string Name { get; }
            public EsiType Type { get; }

            public UnionEntry(string Name, EsiType Type)
            {
                this.Name = Name;
                this.Type = Type;
            }

            public UnionEntry(string Name, Func<EsiType> Type)
            {
                this.Name = Name;
                this.Type = Type();
            }
        }

        public UnionEntry[] Entries { get; }
        public bool IsDiscriminated { get; }

        public EsiUnion(IEnumerable<UnionEntry> Entries, bool IsDiscriminated = true)
            : base()
        {
            this.Entries = Entries.ToArray();
            this.IsDiscriminated = IsDiscriminated;
        }
    }

    public class EsiList : EsiType
    {
        public EsiType Inner { get; }
        public bool IsFixed { get; }

        public EsiList(EsiType Inner, bool IsFixed = true)
            : base()
        {
            this.Inner = Inner;
            this.IsFixed = IsFixed;
        }

        public EsiList(Func<EsiType> Inner, bool IsFixed = true)
            : base()
        {
            this.Inner = Inner();
            this.IsFixed = IsFixed;
        }
    }
}
