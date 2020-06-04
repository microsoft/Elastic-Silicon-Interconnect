using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Capnp;
using CapnpGen;
using Esi.Schema;

namespace Esi.Capnp
{
    /// <summary>
    /// Write a capnp schema message from an ESI schema
    /// </summary>
    public class EsiCapnpWriter : EsiCapnpConvert
    {
        protected Dictionary<EsiObject, UInt64> ObjectToID =
            new Dictionary<EsiObject, ulong>();

        protected List<Node> Nodes = new List<Node>();

            UInt64 ctr = 0xc2c4418e9ca66d10;

        public EsiCapnpWriter(EsiContext ctxt): base(ctxt)
        {    }

        public void Write(EsiSystem sys, FileInfo into)
        {
            if (into.Exists)
                into.Delete();
            using (var stream = into.OpenWrite())
            {
                Write(sys, stream);
            }
        }

        public void Write(EsiSystem sys, Stream stream)
        {
            var cgr = GetCGR(sys);
            var msg = MessageBuilder.Create();
            var cgrRoot = msg.BuildRoot<CodeGeneratorRequest.WRITER>();
            cgr.serialize(cgrRoot);

            var pump = new FramePump(stream);
            pump.Send(msg.Frame);
        }

        private CodeGeneratorRequest GetCGR(EsiSystem sys)
        {
            var objs = sys.Objects;
            AssignIDs(objs);
            return new CodeGeneratorRequest()
            {
                Nodes = GetNodes(objs)
            };
        }

        private void AssignIDs(IEnumerable<EsiObject> objs)
        {
            objs.ForEach(obj =>
            {
                ObjectToID[obj] = AssignId();
            });
        }

        protected ulong AssignId()
        {
            ctr += ctr * ctr;
            return ctr;
        }

        private IReadOnlyList<Node> GetNodes(IEnumerable<EsiObject> objs)
        {
            CreateNames(objs);
            CreateAnnotations();
            objs.ForEach(obj => {
                switch (obj)
                {
                    case EsiStruct st when !string.IsNullOrWhiteSpace(st.Name):
                        CreateStruct(st, ScopeID: null);
                        break;
                    case EsiInterface i:
                        GetNode(i);
                        break;
                }
            });
            return Nodes;
        }

        private void CreateAnnotations()
        {
            EsiCapnpConvert.ESIAnnotations.ForEach(a =>
                Nodes.Add(new Node() {
                    Id = a,
                    DisplayName = Enum.GetName(typeof(AnnotationIDs), a),
                    Annotation = new Node.annotation() {
                        TargetsEnum = true,
                        TargetsStruct = true,
                        TargetsField = true,
                        TargetsUnion = true,
                    }
                })
            );
        }

        private void CreateNames(IEnumerable<EsiObject> objs)
        {
            Nodes.Add(new Node()
            {
                Id = 0xa90c8c4dfeabe4c0,
                NestedNodes = objs.Select(o => o switch
                {
                    EsiNamedType named when !string.IsNullOrWhiteSpace(named.Name) =>
                        new Node.NestedNode()
                        {
                            Id = ObjectToID[named],
                            Name = named.Name
                        },
                    _ => null
                }).Where(n => n != null).ToList(),
            });
            Nodes.Add(new Node()
            {
                Id = 0xb90c8c4dfeabe4c1,
                NestedNodes = EsiCapnpConvert.ESIAnnotations.Select(a =>
                    new Node.NestedNode() {
                        Id = a,
                        Name = Enum.GetName(typeof(EsiCapnpConvert.AnnotationIDs), a),
                    }).Where(n => n != null).ToList(),
            });
        }

        #region ESI Type write methods

        private Node CreateStruct(EsiStruct st, ulong? ScopeID, ushort DataWordCount = 0, ushort PointerCount = 0)
        {
            ulong id = ScopeID.HasValue ? AssignId() : ObjectToID[st];
            if (!ScopeID.HasValue)
                (DataWordCount, PointerCount) = ComputeDPCount(st);
            var n = new Node() {
                Id = id,
                DisplayName = $"struct:{st.Name}",
                ScopeId = ScopeID ?? 0,
                Struct = new Node.@struct() {
                    IsGroup = ScopeID.HasValue,
                    Fields = st.Fields.Select((f, idx) => GetField(f, (ushort)idx, id)).ToList(),
                    DataWordCount = DataWordCount,
                    PointerCount = PointerCount,
                }
            };
            Nodes.Add(n);
            return n;
        }

        private (ushort DataWordCount, ushort PointerCount) ComputeDPCount(EsiStruct st)
        {
            ushort DataWordCount = 0;
            ushort PointerCount = 0;
            st.Traverse(obj => {
                switch (obj) {
                    case EsiReferenceType refType:
                        PointerCount++;
                        break;
                }
            });
            return (DataWordCount, PointerCount);
        }

        private Field GetField(EsiStruct.StructField f, ushort CodeOrder, ulong ScopeID)
        {
            var ret = new Field() {
                Name = f.Name,
                CodeOrder = CodeOrder
            };

            switch (f.Type)
            {
                // *****
                // Special cases of FloatingPoints
                case EsiCompound c when
                    c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                    c.Whole == 8 && c.Fractional == 23:
                    ret.Slot = new Field.slot() {
                        Type = new CapnpGen.Type() {
                            which = CapnpGen.Type.WHICH.Float32,
                        },
                        DefaultValue = new Value() { Float32 = 0 }
                    };
                    break;
                case EsiCompound c when
                    c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                    c.Whole == 11 && c.Fractional == 52:
                    ret.Slot = new Field.slot() {
                        Type = new CapnpGen.Type() {
                            which = CapnpGen.Type.WHICH.Float64
                        },
                        DefaultValue = new Value() { Float64 = 0 }
                    };
                    break;
                // In the general case, refer to a pre-created struct
                case EsiCompound c:
                    ret.Slot = new Field.slot () {
                        Type = new CapnpGen.Type() {
                            Struct = new CapnpGen.Type.@struct() {
                                TypeId = c.Type == EsiCompound.CompoundType.EsiFloat ?
                                    (ulong)AnnotationIDs.FLOATING_POINT_VALUE :
                                    (ulong)AnnotationIDs.FIXED_POINT_VALUE
                            }
                        }
                    };
                    break;

                case EsiStruct st:
                    ret.Group = new Field.group() { TypeId = CreateStruct(st, ScopeID).Id };
                    break;
                case EsiValueType valueType:
                    ret.Slot = new Field.slot() {
                        Type = GetType(valueType),
                        DefaultValue = GetDefaultValue(valueType)
                    };
                    break;
                case EsiReferenceType refType:
                    ret.Slot = new Field.slot() { Type = GetType(refType) };
                    break;
            }

            ret.Annotations = GetAnnotations(f.Type);
            return ret;
        }

        private IReadOnlyList<Annotation> GetAnnotations(EsiType type)
        {
            var ret = new List<Annotation>();
            void Add(AnnotationIDs id, Value v)
            {
                if (v == null)
                    v = new Value () { which = Value.WHICH.Void };
                ret.Add(new Annotation() {
                    Id = (ulong)id,
                    Value = v
                });
            }
            switch (type) {
                case EsiInt i:
                    Add(AnnotationIDs.BITS, new Value() { Uint64 = i.Bits } );
                    break;
                case EsiStruct st:
                    Add(AnnotationIDs.INLINE, null);
                    break;
            };
            return ret.Count == 0 ? null : ret;
        }

        private CapnpGen.Type GetType(EsiValueType valueType)
        {
            switch (valueType) 
            {
                case EsiStruct st:
                    return new CapnpGen.Type() {
                        Struct = new CapnpGen.Type.@struct() { TypeId = ObjectToID[st] }
                    };
                case EsiInt i when i.Bits > 64:
                    C.Log.Warning($"No CapnProto type can fit {i.GetDescriptionTree()}");
                    return null;
                case EsiArray a:
                    C.Log.Warning("There currently does not exist an inline array type in CapnProto");
                    return null;
                case EsiList l:
                    C.Log.Warning("There does not exist an inline list type in CapnProto. Must use a pointer to it.");
                    return null;
                case EsiCompound c:
                    C.Log.Error("Internal error: EsiCompounds are not handled here!");
                    return null;
            }
            return new CapnpGen.Type() {
                which = valueType switch {
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBit => CapnpGen.Type.WHICH.Bool,
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBool => CapnpGen.Type.WHICH.Bool,
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiByte => CapnpGen.Type.WHICH.Uint8,
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiVoid => CapnpGen.Type.WHICH.Void,

                    EsiInt i when i.Signed && i.Bits <= 8 => CapnpGen.Type.WHICH.Int8,
                    EsiInt i when i.Signed && i.Bits <= 16 => CapnpGen.Type.WHICH.Int16,
                    EsiInt i when i.Signed && i.Bits <= 32 => CapnpGen.Type.WHICH.Int32,
                    EsiInt i when i.Signed && i.Bits <= 64 => CapnpGen.Type.WHICH.Int64,

                    EsiInt i when !i.Signed && i.Bits <= 8 => CapnpGen.Type.WHICH.Uint8,
                    EsiInt i when !i.Signed && i.Bits <= 16 => CapnpGen.Type.WHICH.Uint16,
                    EsiInt i when !i.Signed && i.Bits <= 32 => CapnpGen.Type.WHICH.Uint32,
                    EsiInt i when !i.Signed && i.Bits <= 64 => CapnpGen.Type.WHICH.Uint64,
                }
            };
        }

        private Value GetDefaultValue(EsiValueType valueType)
        {
            switch (valueType) 
            {
                case EsiStruct st:
                    return new Value() {
                        Struct = null
                    };
                case EsiInt i when i.Bits > 64:
                    return null;
                case EsiArray a:
                    return null;
                case EsiList l:
                    return null;
                case EsiCompound c:
                    return null;
            }
            return valueType switch {
                EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBit => new Value() { Bool = false },
                EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBool => new Value() { Bool = false },
                EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiByte => new Value() { Uint8 = 0 },
                EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiVoid => new Value() { which = Value.WHICH.Void },
                EsiInt i when i.Signed && i.Bits <= 8 => new Value() { Int8 = 0 },
                EsiInt i when i.Signed && i.Bits <= 16 => new Value() { Int16 = 0 },
                EsiInt i when i.Signed && i.Bits <= 32 => new Value() { Int32 = 0 },
                EsiInt i when i.Signed && i.Bits <= 64 => new Value() { Int64 = 0 },
                EsiInt i when !i.Signed && i.Bits <= 8 => new Value() { Uint8 = 0 },
                EsiInt i when !i.Signed && i.Bits <= 16 => new Value() { Uint16 = 0 },
                EsiInt i when !i.Signed && i.Bits <= 32 => new Value() { Uint32 = 0 },
                EsiInt i when !i.Signed && i.Bits <= 64 => new Value() { Uint64 = 0 },
            };
        }

        private CapnpGen.Type GetType(EsiReferenceType refType)
        {
            switch (refType.Reference)
            {
                case EsiStruct st:
                    return new CapnpGen.Type() {
                        Struct = new CapnpGen.Type.@struct() {
                            TypeId = ObjectToID[st]
                        }
                    };
                case EsiList l when l.Inner is EsiValueType vt:
                    return new CapnpGen.Type() {
                        List = new CapnpGen.Type.list() { ElementType = GetType(vt) }
                    };
                default:
                    throw new EsiCapnpConvertException($"Unsupported ESI type: {refType.GetDescriptionTree()}");
            }
        }

#endregion // ESI Type write methods

        private Node GetNode(EsiInterface iface)
        {
            return null;
        }
    }
}