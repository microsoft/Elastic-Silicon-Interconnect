using System.Diagnostics;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Text;

#nullable enable
namespace Esi.Schema
{
    /// <summary>
    /// void, bool, byte, bit -- datatypes where the width is trivially known
    /// </summary>
    public class EsiPrimitive : EsiTypeParent, EsiValueType
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

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            stringBuilder.Append(Enum.GetName(typeof(PrimitiveType), Type));
        }

        public override string ToString()
        {
            return Enum.GetName(typeof(PrimitiveType), Type);
        }
    }

    /// <summary>
    /// signed and unsigned integers of custom widths
    /// </summary>
    public class EsiInt : EsiTypeParent, EsiValueType
    {
        public bool Signed { get; }
        public ulong Bits { get; }

        public EsiInt(ulong Bits, bool Signed)
            : base()
        {
            this.Bits = Bits;
            this.Signed = Signed;
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            stringBuilder.Append($"{(Signed ? "signed" : "unsigned")} int{Bits}");
        }
    }

    /// <summary>
    /// enum. 'nuff said
    /// </summary>
    public class EsiEnum : EsiTypeParent, EsiValueType
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

        public EnumMember[] Members { get; }

        public EsiEnum(IReadOnlyList<EnumMember> Members)
            : base()
        {
            this.Members = Members.ToArray();
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// fixed and floating point types
    /// 
    /// ideas for a better name accepted
    /// </summary>
    public class EsiCompound : EsiTypeParent, EsiValueType
    {
        public enum CompoundType
        {
            EsiFixed,
            EsiFloat
        }

        public CompoundType Type { get; }
        public bool Signed { get; }
        public ulong Whole { get; }
        public ulong Fractional { get; }

        private static Dictionary<
            (CompoundType Type, bool Signed, ulong Whole, ulong Fractional), EsiCompound> SingletonMapping = 
                new Dictionary<(CompoundType Type, bool Signed, ulong Whole, ulong Fractional), EsiCompound>();
        public static EsiCompound SingletonFor(CompoundType Type, bool Signed, ulong Whole, ulong Fractional)
        {
            var key = (Type, Signed, Whole, Fractional);
            if (!SingletonMapping.TryGetValue(key, out var c))
            {
                c = new EsiCompound(Type, Signed, Whole, Fractional);
                SingletonMapping[key] = c;
            }
            return c;
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            stringBuilder.Append($"{(Signed ? "signed" : "unsigned")} {Enum.GetName(typeof(CompoundType), Type)} {Whole}.{Fractional}");
        }

        private EsiCompound(CompoundType Type, bool Signed, ulong Whole, ulong Fractional)
            : base()
        {
            this.Signed = Signed;
            this.Type = Type;
            this.Whole = Whole;
            this.Fractional = Fractional;
        }
    }

    /// <summary>
    /// a fixed-size array
    /// </summary>
    public class EsiArray : EsiTypeParent, EsiValueType, EsiContainerType
    {
        public EsiType Inner { get; }
        public ulong Length { get; }

        public IEnumerable<EsiType> ContainedTypes => new EsiType[] { Inner };

        public EsiArray(EsiType Inner, ulong Length)
            : base()
        {
            this.Inner = Inner;
            this.Length = Length;
        }

        public EsiContainerType WithInner(EsiType newInner)
        {
            return new EsiArray(newInner, Length);
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            Inner.GetDescriptionTree(stringBuilder, indent);
            stringBuilder.Append($" [{Length}] ");
        }
    }

    /// <summary>
    /// struct: name optional and manditory named fields
    /// </summary>
    public class EsiStruct : EsiTypeParent, EsiValueType, EsiNamedType
    {
        /// <summary>
        /// A struct field
        /// </summary>
        public class StructField : EsiTypeParent
        {
            /// <summary>
            /// field name
            /// </summary>
            public string Name { get; }

            /// <summary>
            /// This is used for versioning in CapnProto. Unsure if it'll be
            /// useful here.
            /// </summary>
            public ulong? CodeOrder { get; }

            /// <summary>
            /// type of the field
            /// </summary>
            public EsiType Type { get; }

            /// <summary>
            /// offset into the struct in bits. BitOffset is into the _entire_ struct.
            /// </summary>
            public int? BitOffset { get; }

            public StructField(string Name, EsiType Type, ulong? CodeOrder = null, int? BitOffset = null)
            {
                this.Name = Name;
                this.Type = Type;
                this.CodeOrder = CodeOrder;
                this.BitOffset = BitOffset;
            }

            public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
            {
                stringBuilder.Indent(indent).Append($"{Name} ");
                Type.GetDescriptionTree(stringBuilder, indent);
                stringBuilder.AppendLine(";");
            }
        }

        public string? Name { get; }
        public StructField[] Fields { get; }
        public readonly IReadOnlyDictionary<string, StructField> FieldLookup;

        public EsiStruct(string? Name, IEnumerable<StructField> Fields)
            : base()
        {
            this.Name = Name;
            this.Fields = Fields.ToArray();
            FieldLookup = this.Fields.ToDictionary(sf => sf.Name, sf => sf);
        }

        public override string ToString()
        {
            return $"struct {Name}";
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            stringBuilder.AppendLine($"{this} {{");
            foreach (var f in Fields)
            {
                f.GetDescriptionTree(stringBuilder, indent+1);
            }
            stringBuilder.Indent(indent).Append("}");
        }
    }

    /// <summary>
    /// An ESI union is like a C union, but can be discriminated like a
    /// SystemVerilog union.
    /// </summary>
    public class EsiUnion : EsiTypeParent, EsiValueType, EsiTypeCollection
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

        public IEnumerable<EsiType> ContainedTypes => Entries.Select(e => e.Type);

        public EsiUnion(IEnumerable<UnionEntry> Entries, bool IsDiscriminated = true)
            : base()
        {
            this.Entries = Entries.ToArray();
            this.IsDiscriminated = IsDiscriminated;
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// A variable-length, ordered collection of items. Similar to a C++ vector.
    /// One key difference is that a 'fixed' list is one whose length is known in
    /// advance of its creation/transmittal so can be sent first. A non-fixed
    /// list is not, so must be ended by an end-of-list token.
    /// </summary>
    public class EsiList : EsiTypeParent, EsiValueType, EsiContainerType
    {
        public EsiType Inner { get; }
        public bool IsFixed { get; }

        public IEnumerable<EsiType> ContainedTypes => new EsiType[] { Inner };

        public EsiList(EsiType Inner, bool IsFixed = true)
            : base()
        {
            Debug.Assert(Inner != null);
            this.Inner = Inner;
            this.IsFixed = IsFixed;
        }

        public EsiContainerType WithInner(EsiType newInner)
        {
            return new EsiList(newInner, IsFixed);
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            Inner.GetDescriptionTree(stringBuilder, indent);
            stringBuilder.Append($" [{(IsFixed ? "fixed" : "")}] ");
        }
    }

    /// <summary>
    /// A reference to another type (typed pointer). Not clear how to implement
    /// in hardware yet. Maybe represents an adjustable offset?
    /// </summary>
    public class EsiReferenceType : EsiTypeParent
    {
        protected Func<EsiType>? Resolver = null;
        protected EsiType? _Reference = null;
        public EsiType? Reference {
            get
            {
                if (_Reference != null)
                    return _Reference;
                if (Resolver != null)
                    return Resolver();
                return null;
            }
            set
            {
                _Reference = value;
            }
        }

        public EsiReferenceType (EsiType Reference)
        {
            this.Reference = Reference;
        }

        public EsiReferenceType(Func<EsiType> Resolver)
        {
            this.Resolver = Resolver;
        }

        public EsiReferenceType WithReference(EsiType newInner)
        {
            return new EsiReferenceType(newInner);
        }

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            switch (Reference)
            {
                case null:
                    stringBuilder.Append(" null ");
                    break;
                case EsiNamedType namedType when (!string.IsNullOrWhiteSpace(namedType.Name)):
                    stringBuilder.Append($" {namedType.Name} * ");
                    break;
                default:
                    Reference.GetDescriptionTree(stringBuilder, indent);
                    stringBuilder.Append(" * ");
                    break;
            }
        }
    }
}
