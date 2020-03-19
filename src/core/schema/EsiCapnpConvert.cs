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


        protected Dictionary<UInt64, EsiCapnpLocation> IDtoNames
            = new Dictionary<ulong, EsiCapnpLocation>();
        protected EsiCapnpLocation GetLocation(UInt64 id)
        {
            if (!IDtoNames.TryGetValue(id, out var loc))
            {
                loc = new EsiCapnpLocation() {
                    Id = id,
                };
                IDtoNames[id] = loc;
            }
            return loc;
        }
        
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

            // First pass: get all the filenames
            var IDtoFile = new Dictionary<ulong, string>();
            cgr.RequestedFiles.Iterate(file => IDtoFile[file.Id] = file.Filename);

            // Second pass: get all the struct names
            cgr.Nodes
                .SelectMany(fileNode => fileNode.NestedNodes.Select(nested => (nested, fileNode)))
                .ForEach(n => 
                    IDtoNames[n.nested.Id] =
                        new EsiCapnpLocation {
                            Id = n.nested.Id,
                            StructName = n.nested.Name,
                            File = IDtoFile.GetValueOrDefault(n.fileNode.Id)
                        });

            // Third pass: get references to each node
            cgr.Nodes.ForEach(n => {
                var loc = GetLocation(n.Id);
                loc.Node = n;
                loc.DisplayName = n.DisplayName;
            });

            // Fourth pass: Do the actual conversion
            var esiTypes = cgr.Nodes.Select(
                node => ConvertNode(node) switch {
                    _ when (ESIAnnotations.Contains(node.Id)) => null,
                    EsiStructReference stRef => stRef.Struct,
                    EsiListReference lstRef => lstRef.List,
                    EsiType t => t,
                    null => null,
            }).Where(t => t != null).ToList();
            return esiTypes;
        }

        /// <summary>
        /// Convert a top-level node to an EsiType, lazily
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        protected EsiType ConvertNode(Node.READER node)
        {
            if (node.Parameters?.Count() > 0 ||
                node.IsGeneric)
            {
                C.Log.Error("Generic types are not (yet?) supported");
                return null;
            }

            var loc = IDtoNames[node.Id];
            // if (string.IsNullOrWhiteSpace(loc.DisplayName) &&
            //     node.ScopeId != 0 &&
            //     IDtoNames.TryGetValue(node.ScopeId, out var scopeLoc))
            // {
            //     // Best effort DisplayName propogation
            //     loc.DisplayName = scopeLoc.DisplayName;
            // }

            switch (node.which)
            {
                case Node.WHICH.Struct:
                    return ConvertStructCached(loc);
                default:
                    return new CapnpEsiErrorType(() => {
                        C.Log.Error(
                            "Type {type} not yet supported. ({loc})",
                            Enum.GetName(typeof(Node.WHICH), node.which),
                            loc);
                    });
            }
        }

        /// <summary>
        /// Return a function which returns an EsiStruct which has been converted from a
        /// CapNProto struct.
        /// </summary>
        private EsiType ConvertStructCached(EsiCapnpLocation loc)
        {
            if (!IDtoType.TryGetValue(loc.Id, out var esiType))
            {
                // First, create a struct reference and populate the cache with it.
                //  This signals that we are descending this struct already, in the case of a cycle
                var stRef = new EsiStructReferenceCapnp((EsiStruct)null);
                esiType = stRef;
                IDtoType[loc.Id] = esiType;

                var canpnStruct = loc.Node.Value.Struct;
                if (canpnStruct.DiscriminantCount == 0) // This capnp struct is not a union
                {
                    var esiStruct = new EsiStruct(
                        Name: loc.StructName,
                        Fields: canpnStruct.Fields.Iterate(f => ConvertField(loc, f))
                    );
                    if (canpnStruct.IsGroup) // This capnp "struct" is actually a group, which is equivalent to an EsiStruct
                    {
                        esiType = esiStruct; // Set the return to the raw struct
                        if (stRef.RefCount > 0) // Check to see that nobody got the tentative reference while we were workin
                            C.Log.Fatal("Found a cycle involving groups. This shouldn't occur! ({loc})", loc);
                        IDtoType[loc.Id] = esiStruct; // And remove the reference to it
                    }
                    else // This capnp "struct" is actually a capnp struct, which is equivalent to an EsiStructReference
                    {
                        stRef.Struct = esiStruct;
                    }
                }
                else // This capnp struct is actually a union
                {
                    C.Log.Error("Unions are not yet supported ({loc})", loc);
                    return null;
                }
            }

            // Mark that we have returned this instance
            if (esiType is EsiStructReferenceCapnp stRefCount)
                stRefCount.RefCount++;
            return esiType;
        }

        /// <summary>
        /// To construct a struct reference, it must exist already in the table
        /// of struct futures.
        /// </summary>
        private EsiType GetNamedNode(UInt64 structId)
        {
            if (IDtoType.TryGetValue(structId, out var esiNamedNode))
                return esiNamedNode;
            
            EsiCapnpLocation loc = null;
            var found = IDtoNames.TryGetValue(structId, out loc);
            switch (found) {
                case true when loc.Node != null:
                    return ConvertNode(loc.Node.Value);
                case true when loc.Node == null:
                case false:
                default:
                    throw new EsiCapnpConvertException($"GetNamedNode failed to find named node {structId}");
            }
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
        private EsiType ConvertType(
            EsiCapnpLocation loc,
            CapnpGen.Type.READER type,
            IReadOnlyList<Annotation.READER> annotations)
        {
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
                CapnpGen.Type.WHICH.Float32 => EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true, 8, 23),
                CapnpGen.Type.WHICH.Float64 => EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true, 11, 52),
                CapnpGen.Type.WHICH.Text => new EsiListReference(new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte), true)),
                CapnpGen.Type.WHICH.Data => new EsiListReference(new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte), true)),

                CapnpGen.Type.WHICH.List => new EsiListReference(new EsiList( ConvertType(loc, type.List.ElementType, null) ) ),
                CapnpGen.Type.WHICH.Enum => GetNamedNode(type.Enum.TypeId),
                CapnpGen.Type.WHICH.Struct => type.Struct.TypeId switch {
                    // ---
                    // "Special", known structs
                    (ulong)AnnotationIDs.FIXED_POINT_VALUE =>
                        EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFixed, true, 63, 64),
                    (ulong)AnnotationIDs.FLOATING_POINT_VALUE =>
                        EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true, 63, 64),

                    // ---
                    // User-specified structs
                    _ => GetNamedNode(type.Struct.TypeId)
                },

                CapnpGen.Type.WHICH.Interface => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the Interface type ({loc})", loc) ),
                CapnpGen.Type.WHICH.AnyPointer => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the AnyPointer type ({loc})", loc) ),

                _ => throw new NotImplementedException($"ConvertType({Enum.GetName(typeof(CapnpGen.Type.WHICH), type.which)}) not implemented ({loc})")
            };
            return AddAnnotations(esiType, loc, annotations);
        }


        /// <summary>
        /// Return a new type based on the old type and the annotation-based modifiers
        /// </summary>
        /// <param name="esiType">The original type</param>
        /// <param name="loc">The original type's Capnp "location"</param>
        /// <param name="annotations">A list of annotations</param>
        /// <returns>The modified EsiType</returns>
        private EsiType AddAnnotations(
            EsiType esiType,
            EsiCapnpLocation loc,
            IReadOnlyList<Annotation.READER> annotations)
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
                    case (EsiStructReference stRef, AnnotationIDs.INLINE) when stRef.Struct != null:
                        esiType = stRef.Struct;
                        break;
                    case (EsiStructReference stRef, AnnotationIDs.INLINE) when stRef.Struct == null:
                        C.Log.Error("$Inline found a data type cycle not broken by a reference type ({loc})", loc);
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
                        esiType = EsiCompound.SingletonFor(
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
                        esiType = EsiCompound.SingletonFor(
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
    /// Extend ESI's struct reference to add ref counting (for internal,
    /// cpnp-specific accounting)
    /// </summary>
    public class EsiStructReferenceCapnp : EsiStructReference
    {
        public long RefCount = 0;

        public EsiStructReferenceCapnp(EsiStruct Struct) : base (Struct)
        {    }

        public EsiStructReferenceCapnp(Func<EsiStruct> Resolver) : base (Resolver)
        {    }
    }

    /// <summary>
    /// Delay an error message until type is used... This may or may not be a good idea.
    /// </summary>
    public class CapnpEsiErrorType : EsiTypeParent
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
    public class EsiCapnpLocation
    {
        public UInt64 Id;
        public Node.READER? Node;
        public string File;
        public string StructName;
        public string DisplayName;
        public IEnumerable<string> Path;

        public EsiCapnpLocation AppendField(string field)
        {
            return new EsiCapnpLocation {
                File = File,
                StructName = StructName,
                DisplayName = DisplayName,
                Path = Path?.Append(field) ?? new string[] { field },
            };
        }

        public override string ToString()
        {
            string fileStruct;
            if (!string.IsNullOrWhiteSpace(DisplayName))
                fileStruct = DisplayName;
            else
                fileStruct = $"{File}:{StructName}";

            if (Path?.Count() > 0)
                return $"{fileStruct}/{string.Join('/', Path)}";
            else
                return fileStruct;
        }
    }

}