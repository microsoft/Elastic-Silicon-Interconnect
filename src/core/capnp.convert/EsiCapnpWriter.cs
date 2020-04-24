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
        protected HashSet<EsiObject> AddlObjects =
            new HashSet<EsiObject>();

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
            CreateImplicit(sys);
            var objs = sys.Objects.Concat(AddlObjects);
            AssignIDs(objs);
            return new CodeGeneratorRequest()
            {
                Nodes = GetNodes(objs)
            };
        }

        private void CreateImplicit(EsiSystem sys)
        {
            sys.Traverse(obj => {
                switch (obj) {
                    case EsiCompound c:
                        AddlObjects.Add(c);
                    break;
                }
            });
            // throw new NotImplementedException();
        }

        private void AssignIDs(IEnumerable<EsiObject> objs)
        {
            UInt64 ctr = 0xc2c4418e9ca66d10;
            objs.ForEach(obj => {
                ctr += ctr * ctr;
                ObjectToID[obj] = ctr;
            });
        }

        private IReadOnlyList<Node> GetNodes(IEnumerable<EsiObject> objs)
        {
            return objs.Select(obj =>
                obj switch {
                    EsiStruct st => GetNode(st),
                    EsiCompound c => GetNode(c),
                    EsiInterface i => GetNode(i),
                }).ToList();
        }

#region ESI Type write methods
        private Node GetNode(EsiCompound c)
        {
            // TODO
            return new Node() {
                Id = ObjectToID[c]
            };
        }

        private Node GetNode(EsiStruct st)
        {
            return new Node() {
                Id = ObjectToID[st],
                Struct = new Node.@struct() {
                    Fields = st.Fields.Select(f => GetField(f)).ToList()
                }
            };
        }

        private Field GetField(EsiStruct.StructField f)
        {
            var ret = new Field() {
                Name = f.Name,
            };

            switch (f.Type)
            {
                // *****
                // Special cases of FloatingPoints
                case EsiCompound c when
                    c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                    c.Whole == 8 && c.Fractional == 23:
                    ret.Slot = new Field.slot() { Type = new CapnpGen.Type() {
                        which = CapnpGen.Type.WHICH.Float32
                    }};
                    break;
                case EsiCompound c when
                    c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                    c.Whole == 11 && c.Fractional == 52:
                    ret.Slot = new Field.slot() { Type = new CapnpGen.Type() {
                        which = CapnpGen.Type.WHICH.Float64
                    }};
                    break;
                // In the general case, refer to a pre-created struct
                case EsiCompound c:
                    ret.Group = new Field.group() { TypeId = ObjectToID[c] };
                    break;

                case EsiStruct st:
                    ret.Group = new Field.group() { TypeId = ObjectToID[st] };
                    break;
                case EsiValueType valueType:
                    ret.Slot = new Field.slot() { Type = GetType(valueType) };
                    break;
                case EsiReferenceType refType:
                    ret.Slot = new Field.slot() { Type = GetType(refType) };
                    break;
            }
            return ret;
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