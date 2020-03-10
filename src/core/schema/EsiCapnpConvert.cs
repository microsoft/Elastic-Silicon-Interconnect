using System.Data.Common;
using System.Diagnostics;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Capnp;
using CapnpGen;
using CliWrap;
using Esi;
using System.Runtime.ExceptionServices;

namespace Esi.Schema 
{
    public class EsiCapnpConvert
    {
        public enum AnnotationIDs : ulong
        {
            BITS = 0xac112269228ad38c,
            INLINE = 0x83f1b26b0188c1bb,
            ARRAY = 0x93ce43d5fd6478ee,
            C_UNION = 0xed2e4e8a596d00a5,
            FIXED_LIST = 0x8e0d4f6349687e9b,
            FIXED = 0xb0aef92d8eed92a5,
            FIXED_POINT = 0x82adb6b7cba4ca97,
            FLOAT = 0xc06dd6e3ee4392de,
            FLOATING_POINT = 0xa9e717a24fd51f71,
            OFFSET = 0xcdbc3408a9217752,
            HWOFFSET = 0xf7afdfd9eb5a7d15,
        }
        public readonly static ISet<ulong> ESIAnnotations = new HashSet<ulong>(
            Enum.GetValues(typeof(AnnotationIDs)).Cast<ulong>());

        public static IReadOnlyList<EsiType> Convert(EsiContext ctxt, CodeGeneratorRequest.READER request)
        {
            var convert = new EsiCapnpConvert(ctxt);
            return convert.GoConvert(request);
        }

        public static IReadOnlyList<EsiType> ConvertFromCGRMessage(EsiContext ctxt, Stream stream)
        {
            var frame = Framing.ReadSegments(stream);
            var deserializer = DeserializerState.CreateRoot(frame);
            var reader = CodeGeneratorRequest.READER.create(deserializer);
            return Convert(ctxt, reader);
        }

        public static IReadOnlyList<EsiType> ConvertTextSchema(EsiContext ctxt, FileInfo file)
        {
            using (var memstream = new MemoryStream() )
            {
                var capnpCmd = Cli.Wrap("capnp")
                    .WithArguments($"compile -I{Path.Join(Esi.Utils.RootDir.FullName, "schema")} -o- {file.FullName}")
                    .WithStandardOutputPipe(PipeTarget.ToStream(memstream))
                    .WithValidation(CommandResultValidation.ZeroExitCode);

                try
                {
                    Task.Run(async () => await capnpCmd.ExecuteAsync()).Wait();
                }
                catch (AggregateException ex)
                {
                    ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                }
                Debug.Assert(memstream.Length > 0);

                memstream.Seek(0, SeekOrigin.Begin);
                var msg = System.Text.Encoding.UTF8.GetString(memstream.ToArray());
                return ConvertFromCGRMessage(ctxt, memstream);
            }
        }


        protected Dictionary<UInt64, string> IDtoFile
            = new Dictionary<ulong, string>();
        protected Dictionary<UInt64, EsiCapnpLocation> IDtoNames
            = new Dictionary<ulong, EsiCapnpLocation>();
        
        protected Dictionary<UInt64, Func<EsiStruct>> IDtoStructFuture
            = new Dictionary<ulong, Func<EsiStruct>>();
        protected Dictionary<UInt64, EsiStruct> IDtoStruct
            = new Dictionary<ulong, EsiStruct>();

        /// <summary>
        /// ESI context member variables are generally called 'C' so it's easier to log stuff
        /// </summary>
        protected EsiContext C;

        public EsiCapnpConvert(EsiContext ctxt)
        {
            this.C = ctxt;
        }
        
        protected IReadOnlyList<EsiType> GoConvert(CodeGeneratorRequest.READER cgr)
        {
            cgr.RequestedFiles.Iterate(file => IDtoFile[file.Id] = file.Filename);
            cgr.Nodes
                .Where(n => n.which == Node.WHICH.File)
                .SelectMany(fileNode => fileNode.NestedNodes.Select(nested => (nested, fileNode)))
                .Iterate(n => 
                    IDtoNames[n.nested.Id] =
                        new EsiCapnpLocation {
                            StructName = n.nested.Name,
                            File = IDtoFile.GetValueOrDefault(n.fileNode.Id)
                        });
            var unresolvedStructs = cgr.Nodes.Select(ConvertNode).Where(t => t != null).ToList();
            var esiTypes = unresolvedStructs.Select(f => f()).ToList();
            return esiTypes;
        }

        protected Func<EsiType> ConvertNode(Node.READER node)
        {
            switch (node.which)
            {
                case Node.WHICH.Struct:
                    return ConvertStruct(node);
            }
            return null;
        }

        private Func<EsiStruct> ConvertStruct(Node.READER s)
        {

            Func<EsiStruct, IEnumerable<EsiStruct.StructField>> GetStructFieldsFuture(EsiCapnpLocation structNameFile)
            {
                return (esiStruct) => {
                    Debug.Assert(!IDtoStruct.ContainsKey(s.Id) ||
                        IDtoStruct[s.Id] == esiStruct );
                    IDtoStruct[s.Id] = esiStruct;
                    return s.Struct.Fields.Select(f => ConvertField(structNameFile, f));
                };
            }
            Func<EsiStruct> stFuture = () => {
                if (!IDtoStruct.TryGetValue(s.Id, out var esiStruct))
                {
                    var structNameFile = IDtoNames.GetValueOrDefault(s.Id);
                    esiStruct = new EsiStruct (
                        Name: structNameFile.StructName,
                        Fields: GetStructFieldsFuture(structNameFile));
                }
                return esiStruct;
            };
            IDtoStructFuture[s.Id] = stFuture;
            return stFuture;
        }

        private Func<EsiStruct> ConvertStruct(UInt64 structId)
        {
            if (IDtoStructFuture.TryGetValue(structId, out var esiStructFuture))
            {
                return esiStructFuture;
            }
            throw new EsiCapnpConvertException($"Future func for struct id {structId} doesn't exist in table!");
        }
        
        private EsiStruct.StructField ConvertField(EsiCapnpLocation structNameFile, Field.READER field)
        {
            switch (field.which)
            {
                case Field.WHICH.Group:
                    return new EsiStruct.StructField(
                        Name: field.Name,
                        Type: ConvertStruct(field.Group.TypeId));
                case Field.WHICH.Slot:
                    return new EsiStruct.StructField(
                        Name: field.Name,
                        Type: ConvertType(
                            structNameFile.AppendField(field.Name),
                            field.Slot.Type,
                            field.Annotations));
                default:
                    throw new EsiCapnpConvertException($"Field type undefined is not a valid capnp schema ({structNameFile})");
            }
        }

        private Func<EsiType> ConvertType(
            EsiCapnpLocation loc,
            CapnpGen.Type.READER type,
            IReadOnlyList<Annotation.READER> annotations)
        {
            return () => {
                var esiType = type.which switch {
                    CapnpGen.Type.WHICH.Void => (EsiType) new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiVoid),
                    CapnpGen.Type.WHICH.Bool => new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiBool),
                    CapnpGen.Type.WHICH.Int8 => new EsiInt(8, true),
                    CapnpGen.Type.WHICH.Int16 => new EsiInt(16, true),
                    CapnpGen.Type.WHICH.Int32 => new EsiInt(32, true),
                    CapnpGen.Type.WHICH.Int64 => new EsiInt(64, true),
                    CapnpGen.Type.WHICH.Uint8 => new EsiInt(8, false),
                    CapnpGen.Type.WHICH.Uint16 => new EsiInt(16, false),
                    CapnpGen.Type.WHICH.Uint32 => new EsiInt(32, false),
                    CapnpGen.Type.WHICH.Uint64 => new EsiInt(64, false),
                    CapnpGen.Type.WHICH.Float32 => new EsiCompound(EsiCompound.CompoundType.EsiFloat, true, 8, 23),
                    CapnpGen.Type.WHICH.Float64 => new EsiCompound(EsiCompound.CompoundType.EsiFloat, true, 11, 52),
                    CapnpGen.Type.WHICH.Text => new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte), true),
                    CapnpGen.Type.WHICH.Data => new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte), true),

                    CapnpGen.Type.WHICH.List => new EsiListReference(new EsiList( ConvertType(loc, type.List.ElementType, null) ) ),
                    CapnpGen.Type.WHICH.Enum => ConvertEnum(loc, type.Enum.TypeId),
                    CapnpGen.Type.WHICH.Struct => new EsiStructReference(ConvertStruct(type.Struct.TypeId)()),

                    CapnpGen.Type.WHICH.Interface => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the Interface type ({loc})", loc) ),
                    CapnpGen.Type.WHICH.AnyPointer => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the AnyPointer type ({loc})", loc) ),

                    _ => throw new NotImplementedException($"ConvertType({Enum.GetName(typeof(CapnpGen.Type.WHICH), type.which)}) not implemented ({loc})")
                };
                return AddAnnotations(esiType, loc, annotations);
            };
        }

        private EsiType ConvertEnum(EsiCapnpLocation loc, ulong typeId)
        {
            throw new NotImplementedException();
        }

        private EsiType AddAnnotations(EsiType esiType, EsiCapnpLocation loc, IReadOnlyList<Annotation.READER> annotations)
        {
            annotations?.ForEach(a => {
                if (!ESIAnnotations.Contains( a.Id ))
                    // No-op if we don't recognize the annotation ID
                    return;

                switch (esiType, (AnnotationIDs) a.Id)
                {
                    case (EsiInt ei, AnnotationIDs.BITS):
                        if (ei.Bits < a.Value.Uint64)
                        {
                            C.Log.Warning(
                                "Specified bits ({SpecifiedBits}) is wider than host type holds ({HostBits})! ({loc})",
                                a.Value.Uint64,
                                ei.Bits,
                                loc);
                        }
                        esiType = new EsiInt(a.Value.Uint64, ei.Signed);
                        break;
                    case (_, AnnotationIDs.BITS):
                        C.Log.Error("$ESI.bits() can only be applied to integer types! ({loc})", loc);
                        break;

                    case (EsiStructReference stRef, AnnotationIDs.INLINE):
                        esiType = stRef.Struct;
                        break;
                    case (EsiListReference lstRef, AnnotationIDs.INLINE):
                        esiType = lstRef.List;
                        break;
                    case (EsiValueType _, AnnotationIDs.INLINE):
                        C.Log.Warning("$inline on value types have no effect ({loc})", loc);
                        break;
                }
            });
            return esiType;
        }
    }
    
    public class CapnpEsiErrorType : EsiType
    {
        public Action A { get; }

        public CapnpEsiErrorType(Action A)
        {
            this.A = A;
        }
    }

    public class EsiCapnpConvertException : Exception
    {
        public EsiCapnpConvertException(string msg) : base (msg) { }
    }

    public struct EsiCapnpLocation
    {
        public string File;
        public string StructName;
        public IReadOnlyList<string> Path;

        public EsiCapnpLocation AppendField(string field)
        {
            return new EsiCapnpLocation {
                File = File,
                Path = Path
            };
        }

        public override string ToString()
        {
            if (Path?.Count() > 0)
                return $"{File}/{StructName}/{string.Join('/', Path)}";
            else
                return $"{File}/{StructName}";
        }
    }

}