using System.Text;
using System.Reflection;
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
using CliWrap.Exceptions;

namespace Esi.Schema 
{
    public class EsiCapnpConvert
    {
        /// <summary>
        /// These are all the annotations which ESI knows about. Keep these in
        /// sync with the ESICoreAnnotations.capnp file.
        /// </summary>
        public enum AnnotationIDs : ulong
        {
            BITS = 0xac112269228ad38c,
            INLINE = 0x83f1b26b0188c1bb,
            ARRAY = 0x93ce43d5fd6478ee,
            C_UNION = 0xed2e4e8a596d00a5,
            FIXED_LIST = 0x8e0d4f6349687e9b,
            FIXED = 0xb0aef92d8eed92a5,
            FIXED_POINT = 0x82adb6b7cba4ca97,
            FIXED_POINT_VALUE = 0x81eebdd3a9e24c9d,
            FLOAT = 0xc06dd6e3ee4392de,
            FLOATING_POINT = 0xa9e717a24fd51f71,
            FLOATING_POINT_VALUE = 0xaf862f0ea103797c,
            OFFSET = 0xcdbc3408a9217752,
            HWOFFSET = 0xf7afdfd9eb5a7d15,
        }
        // Construct a set of all the known Annotations
        public readonly static ISet<ulong> ESIAnnotations = new HashSet<ulong>(
            Enum.GetValues(typeof(AnnotationIDs)).Cast<ulong>());
        
#region EntryPoints
        // ########
        // Various entry points
        //
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
                var errorStringBuilder = new StringBuilder();
                var capnpCmd = Cli.Wrap("capnp")
                    .WithArguments($"compile -I{Path.Join(Esi.Utils.RootDir.FullName, "schema")} -o- {file.FullName}")
                    .WithStandardOutputPipe(PipeTarget.ToStream(memstream))
                    .WithStandardErrorPipe(PipeTarget.ToStringBuilder(errorStringBuilder))
                    .WithValidation(CommandResultValidation.ZeroExitCode);

                try
                {
                    Task.Run(async () => await capnpCmd.ExecuteAsync()).Wait();
                }
                catch (AggregateException ex)
                {
                    if (ex.InnerException is CommandExecutionException cee)
                    {
                        ctxt.Log.Error($"CapNProto Error:\n{errorStringBuilder.ToString()}");
                    }
                    ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                }
                Debug.Assert(memstream.Length > 0);

                memstream.Seek(0, SeekOrigin.Begin);
                return ConvertFromCGRMessage(ctxt, memstream);
            }
        }
#endregion EntryPoints


        protected Dictionary<UInt64, string> IDtoFile
            = new Dictionary<ulong, string>();
        protected Dictionary<UInt64, EsiCapnpLocation> IDtoNames
            = new Dictionary<ulong, EsiCapnpLocation>();
        
        protected Dictionary<UInt64, Func<EsiType>> IDtoTypeFuture
            = new Dictionary<ulong, Func<EsiType>>();
        protected Dictionary<UInt64, EsiType> IDtoType
            = new Dictionary<ulong, EsiType>();

        /// <summary>
        /// ESI context member variables are generally called 'C' so it's easier to log stuff
        /// </summary>
        protected EsiContext C;

        public EsiCapnpConvert(EsiContext ctxt)
        {
            this.C = ctxt;
        }
        
        /// <summary>
        /// Main entry point. Convert a CodeGeneratorRequest to a list of EsiTypes.
        /// </summary>
        /// <param name="cgr"></param>
        /// <returns></returns>
        protected IReadOnlyList<EsiType> GoConvert(CodeGeneratorRequest.READER cgr)
        {
            cgr.RequestedFiles.Iterate(file => IDtoFile[file.Id] = file.Filename);
            cgr.Nodes
                // .Where(n => n.which == Node.WHICH.File)
                .SelectMany(fileNode => fileNode.NestedNodes.Select(nested => (nested, fileNode)))
                .Iterate(n => 
                    IDtoNames[n.nested.Id] =
                        new EsiCapnpLocation {
                            Id = n.nested.Id,
                            StructName = n.nested.Name,
                            File = IDtoFile.GetValueOrDefault(n.fileNode.Id)
                        });
            var unresolvedStructs = cgr.Nodes.Select(ConvertNode).Where(t => t != null).ToList();
            var esiTypes = unresolvedStructs.Select(f => f() switch {
                EsiStructReference stRef => stRef.Struct,
                EsiListReference lstRef => lstRef.List,
                EsiType t => t
            }).ToList();
            return esiTypes;
        }

        /// <summary>
        /// Convert a top-level node to an EsiType, lazily
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        protected Func<EsiType> ConvertNode(Node.READER node)
        {
            if (node.Parameters?.Count() > 0 ||
                node.IsGeneric)
            {
                C.Log.Error("Generic types are not (yet?) supported");
                return null;
            }

            switch (node.which)
            {
                case Node.WHICH.Struct:
                    return ConvertStructCached(node.Id, node.Struct, node.DisplayName);
                default:
                    return () => new CapnpEsiErrorType(() => {
                        C.Log.Error(
                            "Type {type} not yet supported.",
                            Enum.GetName(typeof(Node.WHICH), node.which));
                    });
            }
        }

        /// <summary>
        /// Return a function which returns an EsiStruct which has been converted from a
        /// CapNProto struct.
        ///
        /// This funky, async-y way of doing things is for two reasons:
        ///     1) Capnp schemas contain "pointers" to structs/groups/unions which
        ///     haven't necessarily been read yet (forward-pointers).
        ///     2) These pointers may form cycles. The ESI schema is read-only
        ///     (functional programming style), so the only way to encode cycles into
        ///     it is with "lazy" functions.
        ///
        /// </summary>
        private Func<EsiType> ConvertStructCached(ulong structId, Node.@struct.READER s, string displayName)
        {
            Func<EsiType> stFuture = () => {
                if (!IDtoType.TryGetValue(structId, out var esiType))
                {
                    if (!IDtoNames.TryGetValue(structId, out var structNameFile))
                    {
                        structNameFile = new EsiCapnpLocation {
                            Id = structId,
                            StructName = displayName
                        };
                    }
                    IDtoType[structId] = ConvertStructNow(structNameFile, s);
                }
                return IDtoType[structId];
            };
            IDtoTypeFuture[structId] = stFuture;
            return stFuture;
        }

        /// <summary>
        /// Non-lazy version of ConvertStruct. Should ONLY be called from
        /// ConvertStructCached or if you really, really know what you're doing.
        /// </summary>
        protected EsiType ConvertStructNow(EsiCapnpLocation structContext, Node.@struct.READER s)
        {
            var structId = structContext.Id;

            if (s.DiscriminantCount == 0) // This capnp struct is not a union
            {
                var st = new EsiStruct(
                    Name: structContext.StructName,
                    Fields: (esiStruct) =>
                    {
                        Debug.Assert(!IDtoType.ContainsKey(structId) ||
                            IDtoType[structId] == esiStruct );
                        IDtoType[structId] = esiStruct;
                        return s.Fields.Select(f => ConvertField(structContext, f));
                    });
                if (s.IsGroup) // This capnp "struct" is actually a group, which is equivalent to an EsiStruct
                    return st;
                else // This capnp "struct" is actually a capnp struct, which is equivalent to an EsiStructReference
                    return new EsiStructReference(st);
            }
            else // This capnp struct is actually a union
            {
                C.Log.Error("Unions are not yet supported ({loc})", structContext);
                return null;
            }
        }

        /// <summary>
        /// To construct a struct reference, it must exist already in the table
        /// of struct futures.
        /// </summary>
        private Func<EsiType> GetNamedNode(UInt64 structId)
        {
            if (IDtoTypeFuture.TryGetValue(structId, out var esiStructFuture))
            {
                return esiStructFuture;
            }
            throw new EsiCapnpConvertException($"Future func for struct id {structId} doesn't exist in table!");
        }

        /// <summary>
        /// Convert a struct field which can be either an actual member, "slot", or a group.
        /// </summary>
        private EsiStruct.StructField ConvertField(EsiCapnpLocation structNameFile, Field.READER field)
        {
            switch (field.which)
            {
                case Field.WHICH.Group:
                    return new EsiStruct.StructField(
                        Name: field.Name,
                        Type: GetNamedNode(field.Group.TypeId));

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

        /// <summary>
        /// Entry point for recursion. Should handle any embeddable type and its annotations.
        /// </summary>
        /// <param name="loc"></param>
        /// <param name="type"></param>
        /// <param name="annotations"></param>
        /// <returns></returns>
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
                    CapnpGen.Type.WHICH.Enum => GetNamedNode(type.Enum.TypeId)(),
                    CapnpGen.Type.WHICH.Struct => type.Struct.TypeId switch {
                        // ---
                        // "Special", known structs
                        (ulong)AnnotationIDs.FIXED_POINT_VALUE =>
                            new EsiCompound(EsiCompound.CompoundType.EsiFixed, true, 63, 64),
                        (ulong)AnnotationIDs.FLOATING_POINT_VALUE =>
                            new EsiCompound(EsiCompound.CompoundType.EsiFloat, true, 63, 64),

                        // ---
                        // User-specified structs
                        _ => GetNamedNode(type.Struct.TypeId)()
                    },

                    CapnpGen.Type.WHICH.Interface => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the Interface type ({loc})", loc) ),
                    CapnpGen.Type.WHICH.AnyPointer => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the AnyPointer type ({loc})", loc) ),

                    _ => throw new NotImplementedException($"ConvertType({Enum.GetName(typeof(CapnpGen.Type.WHICH), type.which)}) not implemented ({loc})")
                };
                return AddAnnotations(esiType, loc, annotations);
            };
        }



        /// <summary>
        /// Return a new type based on the old type and the annotation-based modifiers
        /// </summary>
        /// <param name="esiType">The original type</param>
        /// <param name="loc">The original type's Capnp "location"</param>
        /// <param name="annotations">A list of annotations</param>
        /// <returns>The modified EsiType</returns>
        private EsiType AddAnnotations(EsiType esiType, EsiCapnpLocation loc, IReadOnlyList<Annotation.READER> annotations)
        {
            annotations?.ForEach(a => {
                if (!ESIAnnotations.Contains( a.Id ))
                    // No-op if we don't recognize the annotation ID
                    return;

                switch (esiType, (AnnotationIDs) a.Id)
                {
                    // ---
                    // BITS annotation
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

                    // ---
                    // INLINE annotation
                    case (EsiStructReference stRef, AnnotationIDs.INLINE):
                        esiType = stRef.Struct;
                        break;
                    case (EsiListReference lstRef, AnnotationIDs.INLINE):
                        esiType = lstRef.List;
                        break;
                    case (EsiValueType _, AnnotationIDs.INLINE):
                        C.Log.Warning("$inline on value types have no effect ({loc})", loc);
                        break;
                    case (_, AnnotationIDs.INLINE):
                        C.Log.Error("$Inline on '{type}' not expected ({loc})", esiType.GetType(), loc);
                        break;

                    // ---
                    // ARRAY annotation
                    case (EsiListReference listRef, AnnotationIDs.ARRAY):
                        esiType = new EsiArray(listRef.List.Inner, a.Value.Uint64);
                        break;
                    case (EsiList list, AnnotationIDs.ARRAY):
                        esiType = new EsiArray(list.Inner, a.Value.Uint64);
                        break;
                    case (_, AnnotationIDs.ARRAY):
                        C.Log.Error("$Array on '{type}' not valid ({loc})", esiType.GetType(), loc);
                        break;

                    // ---
                    // C_UNION annotation
                    case (_, AnnotationIDs.C_UNION):
                        C.Log.Error("$cUnion not yet supported");
                        break;

                    // ---
                    // FIXED_LIST annotation
                    case (EsiListReference listRef, AnnotationIDs.FIXED_LIST):
                        esiType = new EsiListReference(new EsiList(listRef.List.Inner, true));
                        break;
                    case (EsiList list, AnnotationIDs.FIXED_LIST):
                        esiType = new EsiList(list.Inner, true);
                        break;
                    case (_, AnnotationIDs.FIXED_LIST):
                        C.Log.Error("$FixedList on '{type}' not valid ({loc})", esiType.GetType(), loc);
                        break;

                    // ---
                    // FIXED annotation
                    case (EsiCompound esiCompound, AnnotationIDs.FIXED):
                        var cpnpFixedSpec = new FixedPointSpec.READER(a.Value.Struct);
                        esiType = new EsiCompound(
                            EsiCompound.CompoundType.EsiFixed,
                            cpnpFixedSpec.Signed, cpnpFixedSpec.Whole, cpnpFixedSpec.Fraction);
                        break;
                    case (_, AnnotationIDs.FIXED):
                        C.Log.Error("$Fixed on '{type}' not valid ({loc})", esiType.GetType(), loc);
                        break;

                    // ---
                    // FLOAT annotation
                    case (EsiCompound esiCompound, AnnotationIDs.FLOAT):
                        var cpnpFloatSpec = new FloatingPointSpec.READER(a.Value.Struct);
                        esiType = new EsiCompound(
                            EsiCompound.CompoundType.EsiFloat,
                            cpnpFloatSpec.Signed, cpnpFloatSpec.Exp, cpnpFloatSpec.Mant);
                        break;
                    case (_, AnnotationIDs.FLOAT):
                        C.Log.Error("$Float on '{type}' not valid ({loc})", esiType.GetType(), loc);
                        break;

                    // ---
                    // HWOffset annotation
                    case (_, AnnotationIDs.HWOFFSET):
                        C.Log.Error("$hwoffset not yet supported");
                        break;
                }
            });
            return esiType;
        }
    }
    
    /// <summary>
    /// Delay an error message until type is used... This may or may not be a good idea.
    /// </summary>
    public class CapnpEsiErrorType : EsiType
    {
        public Action A { get; }

        public CapnpEsiErrorType(Action A)
        {
            this.A = A;
        }
    }

    /// <summary>
    /// An exception in the conversion process
    /// </summary>
    public class EsiCapnpConvertException : Exception
    {
        public EsiCapnpConvertException(string msg) : base (msg) { }
    }

    /// <summary>
    /// Track various data and metadata about capnp types
    /// </summary>
    public struct EsiCapnpLocation
    {
        public UInt64 Id;
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