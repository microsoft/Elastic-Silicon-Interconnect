using System.Diagnostics;
using System.Threading;
using Microsoft.Win32.SafeHandles;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

#nullable enable
namespace Esi.Schema
{

    public interface EsiObject
    {
        void GetDescriptionTree(StringBuilder stringBuilder, uint indent);
        public string GetDescriptionTree(uint indent=0)
        {
            var sb = new StringBuilder();
            GetDescriptionTree(sb, indent);
            return sb.ToString();
        }
    }

    public class EsiSystem
    {
        public IEnumerable<EsiObject> Objects { get; }
        public IReadOnlyDictionary<string, EsiNamedType> NamedTypes { get; }


        public EsiSystem (IEnumerable<EsiObject> Objects)
        {
            this.Objects = Objects.ToArray();
            NamedTypes =
                this.Objects
                    .Select(t => t as EsiNamedType)
                    .Where(t => t != null && t.Name != null)
                    .ToDictionary(t => t?.Name!, t => t!);
        }
    }


    public interface EsiType : EsiObject
    {
        bool StructuralEquals(EsiType that, IDictionary<EsiType, EsiType?>? objMap = null);
    }
    
    public interface EsiNamedType : EsiType
    {
        string? Name { get; }
    }

    public interface EsiValueType : EsiType
    {    }

    public interface EsiContainerType : EsiType
    {
        EsiType Inner { get; }

        EsiContainerType WithInner(EsiType newInner);
    }

    public abstract partial class EsiTypeParent : EsiType
    {
        public abstract void GetDescriptionTree(StringBuilder stringBuilder, uint indent);
        public string GetDescriptionTree(uint indent=0)
        {
            return ((EsiObject)this).GetDescriptionTree(indent);
        }
    }

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

    public class EsiArray : EsiTypeParent, EsiValueType, EsiContainerType
    {
        public EsiType Inner { get; }
        public ulong Length { get; }

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

    public class EsiStruct : EsiTypeParent, EsiValueType, EsiNamedType
    {
        public class StructField : EsiTypeParent
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

        /// <summary>
        /// UPDATE: I found a better way to do this. Keeping this here in case it
        /// becomes useful again in the future.
        ///
        /// This constructor is somewhat unintuitive, but is useful
        /// to encode cycles in this read-only, functional style object schema.
        /// 
        /// By having to call 'Fields' with a reference to 'this' (which is not
        /// available before the construction), this instance can be used in its
        /// 'StructFields', directly or indirectly.
        /// 
        /// </summary>
        // public EsiStruct(string? Name, Func<EsiStruct, IEnumerable<StructField>> Fields)
        //     : base()
        // {
        //     this.Name = Name;
        //     this.Fields = Fields(this).ToArray();
        //     FieldLookup = this.Fields.ToDictionary(sf => sf.Name, sf => sf);
        // }

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

    public class EsiUnion : EsiTypeParent, EsiValueType
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

        public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            throw new NotImplementedException();
        }
    }

    public class EsiList : EsiTypeParent, EsiValueType, EsiContainerType
    {
        public EsiType Inner { get; }
        public bool IsFixed { get; }

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
