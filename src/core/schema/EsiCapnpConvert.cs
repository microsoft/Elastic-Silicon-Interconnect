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

namespace Esi.Schema 
{
    public class EsiCapnpConvert
    {
        public struct AnnotationIDs
        {
            public const ulong BITS = 0xac112269228ad38c;
        }

        public static IReadOnlyList<EsiType> Convert(CodeGeneratorRequest request)
        {
            var convert = new EsiCapnpConvert();
            return convert.GoConvert(request);
        }

        public static IReadOnlyList<EsiType> ConvertFromCGRMessage(Stream stream)
        {
            var frame = Framing.ReadSegments(stream);
            var deserializer = DeserializerState.CreateRoot(frame);
            var reader = CodeGeneratorRequest.READER.create(deserializer);
            var cgr = new CodeGeneratorRequest();
            cgr.Nodes = reader.Nodes.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.Node>(_));
            cgr.RequestedFiles = reader.RequestedFiles.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.CodeGeneratorRequest.RequestedFile>(_));
            cgr.CapnpVersion = CapnpSerializable.Create<CapnpGen.CapnpVersion>(reader.CapnpVersion);
            cgr.applyDefaults();
            return Convert(cgr);
        }

        public static IReadOnlyList<EsiType> ConvertTextSchema(FileInfo file)
        {
            using (var memstream = new MemoryStream() )
            {
                var capnpCmd = Cli.Wrap("capnp")
                    .WithArguments($"compile -I{Path.Join(Esi.Utils.RootDir.FullName, "schema")} -o- {file.FullName}")
                    .WithStandardOutputPipe(PipeTarget.ToStream(memstream))
                    .WithValidation(CommandResultValidation.ZeroExitCode);

                Task.Run(async () => await capnpCmd.ExecuteAsync()).Wait();
                Debug.Assert(memstream.Length > 0);

                memstream.Seek(0, SeekOrigin.Begin);
                return ConvertFromCGRMessage(memstream);
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
                        Type: ConvertType(field.Slot.Type));
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

        private Func<EsiType> ConvertType(CapnpGen.Type type)
        {
            if (SimpleTypeMappings.TryGetValue(type.which, out var esiType))
            {
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