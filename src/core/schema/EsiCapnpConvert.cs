using System.Net.WebSockets;
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
        public struct AnnotationIDs
        {
            public const ulong BITS = 0xac112269228ad38c;
            public const ulong INLINE = 0x83f1b26b0188c1bb;
            public const ulong ARRAY = 0x93ce43d5fd6478ee;
            public const ulong C_UNION = 0xed2e4e8a596d00a5;
            public const ulong FIXED_LIST = 0x8e0d4f6349687e9b;
            public const ulong FIXED = 0xb0aef92d8eed92a5;
            public const ulong FIXED_POINT = 0x82adb6b7cba4ca97;
            public const ulong FLOAT = 0xc06dd6e3ee4392de;
            public const ulong FLOATING_POINT = 0xa9e717a24fd51f71;
            public const ulong OFFSET = 0xcdbc3408a9217752;
            public const ulong HWOFFSET = 0xf7afdfd9eb5a7d15;
        }

        public static IReadOnlyList<EsiType> Convert(EsiContext ctxt, CodeGeneratorRequest request)
        {
            var convert = new EsiCapnpConvert(ctxt);
            return convert.GoConvert(request);
        }

        public static IReadOnlyList<EsiType> ConvertFromCGRMessage(EsiContext ctxt, Stream stream)
        {
            var frame = Framing.ReadSegments(stream);
            var deserializer = DeserializerState.CreateRoot(frame);
            var reader = CodeGeneratorRequest.READER.create(deserializer);
            var cgr = new CodeGeneratorRequest();
            cgr.Nodes = reader.Nodes.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.Node>(_));
            cgr.RequestedFiles = reader.RequestedFiles.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.CodeGeneratorRequest.RequestedFile>(_));
            cgr.CapnpVersion = CapnpSerializable.Create<CapnpGen.CapnpVersion>(reader.CapnpVersion);
            cgr.applyDefaults();
            return Convert(ctxt, cgr);
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
        protected Dictionary<UInt64, (string name, string file)> IDtoNames
            = new Dictionary<ulong, (string name, string file)>();
        
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
        
        protected IReadOnlyList<EsiType> GoConvert(CodeGeneratorRequest cgr)
        {
            cgr.RequestedFiles.Iterate(file => IDtoFile[file.Id] = file.Filename);
            cgr.Nodes
                .Where(n => n.which == Node.WHICH.File)
                .SelectMany(fileNode => fileNode.NestedNodes.Select(nested => (nested, fileNode)))
                .Iterate(n => 
                    IDtoNames[n.nested.Id] =
                        (name: n.nested.Name, file: IDtoFile.GetValueOrDefault(n.fileNode.Id)));
            var unresolvedStructs = cgr.Nodes.Select(ConvertNode).Where(t => t != null).ToList();
            var esiTypes = unresolvedStructs.Select(f => f()).ToList();
            return esiTypes;
        }

        protected Func<EsiType> ConvertNode(Node node)
        {
            switch (node.which)
            {
                case Node.WHICH.Struct:
                    return ConvertStruct(node);
            }
            return null;
        }

        private Func<EsiStruct> ConvertStruct(Node s)
        {

            Func<EsiStruct, IEnumerable<EsiStruct.StructField>> GetStructFieldsFuture()
            {
                return (esiStruct) => {
                    Debug.Assert(!IDtoStruct.ContainsKey(s.Id) ||
                        IDtoStruct[s.Id] == esiStruct );
                    IDtoStruct[s.Id] = esiStruct;
                    return s.Struct.Fields.Select(ConvertField);
                };
            }
            Func<EsiStruct> stFuture = () => {
                if (!IDtoStruct.TryGetValue(s.Id, out var esiStruct))
                {
                    esiStruct = new EsiStruct (
                        Name: IDtoNames.GetValueOrDefault(s.Id).name,
                        Fields: GetStructFieldsFuture());
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
        
        private EsiStruct.StructField ConvertField(Field field)
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
                        Type: ConvertType(field.Slot.Type, field.Annotations));
                default:
                    throw new EsiCapnpConvertException("Field type undefined is not a valid capnp schema");
            }
        }

        public IReadOnlyDictionary<CapnpGen.Type.WHICH, EsiType> SimpleTypeMappings =
            new Dictionary<CapnpGen.Type.WHICH, EsiType> {
                [CapnpGen.Type.WHICH.Void] = new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiVoid),
                [CapnpGen.Type.WHICH.Bool] = new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiBool),
                [CapnpGen.Type.WHICH.Int8] = new EsiInt(8, true),
                [CapnpGen.Type.WHICH.Int16] = new EsiInt(16, true),
                [CapnpGen.Type.WHICH.Int32] = new EsiInt(32, true),
                [CapnpGen.Type.WHICH.Int64] = new EsiInt(64, true),
                [CapnpGen.Type.WHICH.Uint8] = new EsiInt(8, false),
                [CapnpGen.Type.WHICH.Uint16] = new EsiInt(16, false),
                [CapnpGen.Type.WHICH.Uint32] = new EsiInt(32, false),
                [CapnpGen.Type.WHICH.Uint64] = new EsiInt(64, false),
                [CapnpGen.Type.WHICH.Float32] = new EsiCompound(EsiCompound.CompoundType.EsiFloat, true, 8, 23),
                [CapnpGen.Type.WHICH.Float64] = new EsiCompound(EsiCompound.CompoundType.EsiFloat, true, 11, 52),
                [CapnpGen.Type.WHICH.Text] = new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte), true),
                [CapnpGen.Type.WHICH.Data] = new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte), true),

            };

        private Func<EsiType> ConvertType(CapnpGen.Type type, IReadOnlyList<Annotation> annotations)
        {
            if (SimpleTypeMappings.TryGetValue(type.which, out var esiType))
            {
                if (annotations?.Count() > 0)
                {
                    foreach (var a in annotations)
                    {
                        switch (a.Id)
                        {
                            case AnnotationIDs.BITS:
                                if (esiType is EsiInt ei)
                                {
                                    if (ei.Bits < a.Value.Uint64)
                                    {
                                        C.Log.Warning(
                                            "Specified bits ({SpecifiedBits}) is wider than host type holds ({HostBits})!",
                                            a.Value.Uint64.Value,
                                            ei.Bits);
                                    }
                                    return () => new EsiInt(a.Value.Uint64.Value, ei.Signed);
                                }
                                else
                                {
                                    C.Log.Error("$ESI.bits() can only be applied to integer types!");
                                }
                                break;
                        }
                    }
                }
                return () => esiType;
            }

            switch (type.which)
            {
                case CapnpGen.Type.WHICH.Struct:
                    return ConvertStruct(type.Struct.TypeId);
                default:
                    throw new NotImplementedException($"ConvertType({Enum.GetName(typeof(CapnpGen.Type.WHICH), type.which)}) not implemented");
            }
        }
    }
    
    public class EsiCapnpConvertException : Exception
    {
        public EsiCapnpConvertException(string msg) : base (msg) { }
    }

}