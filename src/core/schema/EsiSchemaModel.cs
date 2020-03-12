using Microsoft.Win32.SafeHandles;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Reflection;

#nullable enable
namespace Esi.Schema
{
    public abstract partial class EsiType
    {    }

    public class EsiValueType : EsiType
    {    }

    public class EsiPrimitive : EsiValueType
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

    public class EsiInt : EsiValueType
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

    public class EsiEnum : EsiValueType
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
    }

    public class EsiCompound : EsiValueType
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

        public EsiCompound(CompoundType Type, bool Signed, ulong Whole, ulong Fractional)
            : base()
        {
            this.Type = Type;
            this.Whole = Whole;
            this.Fractional = Fractional;
        }
    }

    public class EsiArray : EsiValueType
    {
        public EsiType Inner { get; }
        public ulong Length { get; }

        public EsiArray(EsiType Inner, ulong Length)
            : base()
        {
            this.Inner = Inner;
            this.Length = Length;
        }

        public EsiArray(Func<EsiType> Inner, ulong Length)
            : base()
        {
            this.Inner = Inner();
            this.Length = Length;
        }
    }

    public class EsiStruct : EsiValueType
    {
        public class StructField : EsiType
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
        protected IReadOnlyDictionary<string, StructField> FieldLookup { get; }

        public EsiStruct(string? Name, IEnumerable<StructField> Fields)
            : base()
        {
            this.Name = Name;
            this.Fields = Fields.ToArray();
            FieldLookup = this.Fields.ToDictionary(sf => sf.Name, sf => sf);
        }

        /// <summary>
        /// This constructor is somewhat unintuitive, but is useful (necessary?)
        /// to encode cycles in this read-only, functional style object schema.
        /// 
        /// By having to call 'Fields' with a reference to 'this' (which is not
        /// available before the construction), this instance can be used in its
        /// 'StructFields', directly or indirectly.
        /// 
        /// </summary>
        public EsiStruct(string? Name, Func<EsiStruct, IEnumerable<StructField>> Fields)
            : base()
        {
            this.Name = Name;
            this.Fields = Fields(this).ToArray();
            FieldLookup = this.Fields.ToDictionary(sf => sf.Name, sf => sf);
        }
    }

    public class EsiStructReference : EsiType 
    {
        protected Func<EsiStruct>? Resolver = null;
        protected EsiStruct? _Struct = null;
        public EsiStruct? Struct {
            get
            {
                if (_Struct != null)
                    return _Struct;
                if (Resolver != null)
                    return Resolver();
                return null;
            }
            set
            {
                _Struct = value;
            }
        }

        public EsiStructReference (EsiStruct Struct)
        {
            this.Struct = Struct;
        }

        public EsiStructReference(Func<EsiStruct> Resolver)
        {
            this.Resolver = Resolver;
        }
    }

    public class EsiUnion : EsiValueType
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

    public class EsiList : EsiValueType
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

    public class EsiListReference : EsiType
    {
        public EsiList List { get; set; }

        public EsiListReference(EsiList List)
        {
            this.List = List;
        }
    }
}
