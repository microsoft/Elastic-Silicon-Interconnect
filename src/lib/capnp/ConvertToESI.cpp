// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "capnp/CapnpConvert.hpp"
#include "Dialects/Esi/EsiTypes.hpp"
#include <llvm/Support/FormatVariadic.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/StandardTypes.h>

#include <vector>
#include <memory>
#include <map>
#include <list>
#include <cstdint>
#include <printf.h>
#include <deque>

using namespace std;
using namespace mlir::esi;
using namespace capnp;
using namespace capnp::schema;

namespace esi {
namespace capnp {

class CapnpParser {

    /// <summary>
    /// Track various data and metadata about capnp types
    /// </summary>
    struct EsiCapnpLocation
    {
    public:
        ParsedSchema node;
        mlir::Type type;
        kj::StringPtr nodeName;
        string file;
        list<string> path;

        // EsiCapnpLocation(const EsiCapnpLocation& other) :
        //     node(other.node), type(other.type), nodeName(other.nodeName), file(other.file), displayName(other.displayName), path(other.path) {
        // }

        // EsiCapnpLocation(Node::Reader* node, string name, string file) :
        //     node(node), nodeName(name), file(file), displayName(node->getDisplayName()) { }

        EsiCapnpLocation AppendField(string field)
        {
            EsiCapnpLocation ret = *this;
            ret.path.push_back(field);
            return ret;
        }

        string ToString()
        {
            string fileStruct;
            auto displayName = node.getProto().getDisplayName();
            if (!all_of(displayName.begin(), displayName.end(), iswspace))
                fileStruct = displayName;
            else
                fileStruct = llvm::formatv("{0}:{1}", file, nodeName);

            if (path.size() > 0)
                return llvm::formatv("{0}/{1}", fileStruct, llvm::join(path, ","));
            else
                return fileStruct;
        }
    };

    EsiCapnpLocation& GetLocation(ParsedSchema node) {
        auto id = node.getProto().getId();
        auto locIter = IDtoLoc.find(id);
        if (locIter == IDtoLoc.end()) {
            EsiCapnpLocation loc = {
                .node = node,
                .nodeName = node.getNodeName(),
            };
            // EsiCapnpLocation(&node, IDtoNames[id], IDtoFile[id]);
            IDtoLoc[id] = loc;
        }
        return IDtoLoc[id];
    }

    std::map<uint64_t, string> IDtoFile;
    std::map<uint64_t, std::string> IDtoNames;
    std::map<uint64_t, EsiCapnpLocation> IDtoLoc;
    mlir::MLIRContext* ctxt;

public:
    CapnpParser(mlir::MLIRContext* ctxt) :
        ctxt(ctxt) { }

    llvm::Error ConvertTypes(kj::Array<ParsedSchema> nodes, std::vector<mlir::Type>& outputTypes) {
        // First pass: get all the filenames
        // for (auto file : cgr.getRequestedFiles()) {
            // IDtoFile[file.getId()] = file.getFilename();
        // }

        // Second pass: get all the node names
        // for (auto node : nodes) {
        //     for (auto nestedNode : node.getNestedNodes()) {
        //         IDtoNames[nestedNode.getId()] = nestedNode.getName();
        //     }
        // }

        for (auto node : nodes) {
            auto& nodeLoc = GetLocation(node);

            if (node.getProto().isStruct()) {
                auto rc = ConvertStruct(nodeLoc);
                if (mlir::failed(rc)) {
                    llvm::errs() << llvm::formatv("Failed to convert '{0}' to EsiStruct\n", node.getProto().getDisplayName());
                }
            } else {
                llvm::errs() << llvm::formatv("Unconvertable type (node '{0}')\n", node.getProto().getDisplayName());
            }
        }

        for (auto locIdPair : IDtoLoc) {
            outputTypes.push_back(locIdPair.second.type);
        }

        return llvm::Error::success();
    }

    //     /// <summary>
    //     /// Main entry point. Convert a CodeGeneratorRequest to a list of EsiTypes.
    //     /// </summary>
    //     /// <param name="cgr"></param>
    //     /// <returns></returns>
    //     protected EsiSystem Read(CodeGeneratorRequest.READER cgr)
    //     {
    //         ulong CapnpSchemaID = cgr.RequestedFiles.FirstOrDefault().Id;

    //         return sys;
    //     }

    /// <summary>
    /// Convert a top-level node to an EsiType, lazily
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    mlir::LogicalResult ConvertStruct(EsiCapnpLocation& loc)
    {
        const auto& node = loc.node;
        const auto& _struct = node.asStruct();
        const auto& fields = _struct.getFields();
        const size_t num = fields.size();

        vector<MemberInfo> esiFields(num);
        for (auto i=0; i<num; i++) {
            auto rc = ConvertField(
                loc,
                fields[i],
                esiFields[i]);
        }
        loc.type = StructType::get(ctxt, esiFields);
        return mlir::success();
    }

    //     private EsiInterface ConvertInterface(ulong id)
    //     {
    //         var loc = IDtoNames[id];
    //         if (!loc.Node.HasValue)
    //             throw new EsiCapnpConvertException($"Could not find node for interface {loc}");
    //         var iface = loc.Node.Value.Interface;
    //         var superClasses = iface.Superclasses.Select(s => ConvertInterface(s.Id)).ToArray();
    //         return new EsiInterface(
    //             Name: loc.NodeName,
    //             Methods: iface.Methods.Select(ConvertMethod).ToArray()
    //         );
    //     }

    //     private EsiInterface.Method ConvertMethod(Method.READER method)
    //     {
    //         IEnumerable<(string Name, EsiType type)> MethodConvert(EsiType t)
    //         {
    //             if (!(t is EsiReferenceType refType))
    //                 throw new EsiCapnpConvertException($"Internal error: expected reference, got {t.GetType()}");

    //             t = refType.Reference;
    //             if (!(t is EsiStruct st))
    //                 throw new EsiCapnpConvertException($"Internal error: expected struct reference, got {t.GetType()}*");

    //             return st.Fields.Select(f =>
    //                 (Name: f.Name,
    //                 Type: f.Type is EsiReferenceType refType ?
    //                                 refType.Reference : f.Type));
    //         }

    //         if (method.ResultBrand.Scopes.Count() > 0 ||
    //             method.ParamBrand.Scopes.Count() > 0)
    //             C.Log.Error("Generics currently unsupported");
    //         return new EsiInterface.Method(
    //             Name: method.Name,
    //             Params: MethodConvert(GetNamedType(method.ParamStructType)),
    //             Returns: MethodConvert(GetNamedType(method.ResultStructType))
    //         );
    //     }

    //     /// <summary>
    //     /// Return a function which returns an EsiStruct which has been converted from a
    //     /// CapNProto struct.
    //     /// </summary>
    //     private EsiType ConvertStructCached(ulong id)
    //     {
    //         var loc = IDtoNames[id];
    //         if (!IDtoType.TryGetValue(loc.Id, out var esiType))
    //         {
    //             // First, create a struct reference and populate the cache with it.
    //             //  This signals that we are descending this struct already, in the case of a cycle
    //             var stRef = new EsiReferenceCapnp((EsiStruct)null);
    //             esiType = stRef;
    //             IDtoType[loc.Id] = esiType;

    //             var canpnStruct = loc.Node.Value.Struct;
    //             if (canpnStruct.DiscriminantCount == 0) // This capnp struct is not a union
    //             {
    //                 var esiStruct = new EsiStruct(
    //                     Name: loc.NodeName,
    //                     Fields: canpnStruct.Fields.Iterate(f => ConvertField(loc, f))
    //                 );
    //                 if (canpnStruct.IsGroup) // This capnp "struct" is actually a group, which is equivalent to an EsiStruct
    //                 {
    //                     esiType = esiStruct; // Set the return to the raw struct
    //                     if (stRef.RefCount > 0) // Check to see that nobody got the tentative reference while we were workin
    //                         C.Log.Fatal("Found a cycle involving groups. This shouldn't occur! ({loc})", loc);
    //                     IDtoType[loc.Id] = esiStruct; // And remove the reference to it
    //                 }
    //                 else // This capnp "struct" is actually a capnp struct, which is equivalent to an EsiStructReference
    //                 {
    //                     stRef.Reference = esiStruct;
    //                 }
    //             }
    //             else // This capnp struct is actually a union
    //             {
    //                 C.Log.Error("Unions are not yet supported ({loc})", loc);
    //                 return null;
    //             }
    //         }

    //         // Mark that we have returned this instance
    //         if (esiType is EsiReferenceCapnp stRefCount)
    //             stRefCount.RefCount++;
    //         if (esiType is EsiType ty)
    //             return ty;
    //         C.Log.Error("Unsupported type: {type}", esiType.GetType());
    //         return null;
    //     }

    //     /// <summary>
    //     /// To construct a struct reference, it must exist already in the table
    //     /// of struct futures.
    //     /// </summary>
    //     private EsiObject GetNamedNode(UInt64 structId)
    //     {
    //         if (IDtoType.TryGetValue(structId, out var esiNamedNode))
    //             return esiNamedNode;
            
    //         EsiCapnpLocation loc = null;
    //         var found = IDtoNames.TryGetValue(structId, out loc);
    //         switch (found) {
    //             case true when loc.Node != null:
    //                 return ConvertNode(loc.Node.Value);
    //             case true when loc.Node == null:
    //             case false:
    //             default:
    //                 throw new EsiCapnpConvertException($"GetNamedNode failed to find named node {structId}");
    //         }
    //     }

    //     private EsiType GetNamedType(UInt64 structId)
    //     {
    //         var esiObj = GetNamedNode(structId);
    //         if (esiObj is EsiType type)
    //             return type;
    //         C.Log.Error("Unsupported use as data type: {type}", esiObj?.GetType());
    //         return null;
    //     }

    /// <summary>
    /// Convert a struct field which can be either an actual member, "slot", or a group.
    /// </summary>
    mlir::LogicalResult ConvertField(EsiCapnpLocation loc, StructSchema::Field field, MemberInfo& mi)
    {
        auto fieldLoc = loc.AppendField(field.getProto().getName());
        switch (field.getProto().which())
        {
            // case Field::Which::GROUP:
            //     mi = {
            //         .name = field.getName(),
            //         .type = AddAnnotations(
            //             ConvertStruct(fieldLoc),
            //             field.getAnnotations()
            //         )
            //     };
            //     return mlir::success();

            case Field::Which::SLOT:
                mi.name = field.getProto().getName();
                return ConvertType(fieldLoc, field.getType(), field.getProto().getAnnotations(), mi);
            default:
                return mlir::failure();
        }
    }

    /// <summary>
    /// Entry point for recursion. Should handle any embeddable type and its annotations.
    /// </summary>
    /// <param name="loc"></param>
    /// <param name="type"></param>
    /// <param name="annotations"></param>
    /// <returns></returns>
    mlir::LogicalResult ConvertType(
        EsiCapnpLocation loc,
        ::capnp::Type type,
        ::capnp::List< ::capnp::schema::Annotation,  ::capnp::Kind::STRUCT>::Reader annotations,
        MemberInfo& mi)
    {
        switch (type.which()) {
            case ::capnp::schema::Type::Which::VOID:
                mi.set(mlir::NoneType::get(ctxt));
                break;
            case ::capnp::schema::Type::Which::BOOL:
                mi.set(mlir::IntegerType::get(1, mlir::IntegerType::SignednessSemantics::Signless, ctxt));
                break;
            case ::capnp::schema::Type::Which::INT8:
                mi.set(mlir::IntegerType::get(8, mlir::IntegerType::SignednessSemantics::Signed, ctxt));
                break;
            case ::capnp::schema::Type::Which::INT16:
                mi.set(mlir::IntegerType::get(16, mlir::IntegerType::SignednessSemantics::Signed, ctxt));
                break;
            case ::capnp::schema::Type::Which::INT32:
                mi.set(mlir::IntegerType::get(32, mlir::IntegerType::SignednessSemantics::Signed, ctxt));
                break;
            case ::capnp::schema::Type::Which::INT64:
                mi.set(mlir::IntegerType::get(64, mlir::IntegerType::SignednessSemantics::Signed, ctxt));
                break;
            case ::capnp::schema::Type::Which::UINT8:
                mi.set(mlir::IntegerType::get(8, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt));
                break;
            case ::capnp::schema::Type::Which::UINT16:
                mi.set(mlir::IntegerType::get(16, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt));
                break;
            case ::capnp::schema::Type::Which::UINT32:
                mi.set(mlir::IntegerType::get(32, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt));
                break;
            case ::capnp::schema::Type::Which::UINT64:
                mi.set(mlir::IntegerType::get(64, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt));
                break;
            // case Type::Which::FLOAT32:
            //     type = EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true, 8, 23),
            //     break;
            // case Type::Which::FLOAT64:
            //     type = EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true, 11, 52),
            //     break;
            // case Type::Which::TEXT:
            //     type = new EsiReferenceType(new EsiList(EsiPrimitive.Byte, true)),
            //     break;
            // case Type::Which::DATA:
            //     type = new EsiReferenceType(new EsiList(EsiPrimitive.Byte, true)),
            //     break;
            // case Type::Which::LIST:
            //     type = new EsiReferenceType(new EsiList( ConvertType(loc, type.List.ElementType, null) ) ),
            //     break;
            // case Type::Which::ENUM:
            //     type = GetNamedType(type.Enum.TypeId),
            //     break;
            // case Type::Which::STRUCT:
            //     mi.type = type.Struct.TypeId switch {
            //         break;
            //     // ---
            //     // "Special", known structs
            //     (ulong)AnnotationIDs.FIXED_POINT_VALUE =>
            //         EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFixed, true, 63, 64),
            //     (ulong)AnnotationIDs.FLOATING_POINT_VALUE =>
            //         EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true, 63, 64),

            //     // ---
            //     // User-specified structs
            //     _ => GetNamedType(type.Struct.TypeId)
            // },

            // CapnpGen.Type.WHICH.Interface => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the Interface type ({loc})", loc) ),
            // CapnpGen.Type.WHICH.AnyPointer => new CapnpEsiErrorType( () => C.Log.Error("ESI does not support the AnyPointer type ({loc})", loc) ),

            // _ => throw new NotImplementedException($"ConvertType({Enum.GetName(typeof(CapnpGen.Type.WHICH), type.which)}) not implemented ({loc})")
            default:
                llvm::errs() << llvm::formatv("Capnp type number {0} not supported (at {1})\n", type.which(), loc.ToString());
                return mlir::failure();
        };
        return mlir::success();
        // return AddAnnotations(mi, loc, annotations);
    }


    //     /// <summary>
    //     /// Return a new type based on the old type and the annotation-based modifiers
    //     /// </summary>
    //     /// <param name="esiType">The original type</param>
    //     /// <param name="loc">The original type's Capnp "location"</param>
    //     /// <param name="annotations">A list of annotations</param>
    //     /// <returns>The modified EsiType</returns>
    //     private EsiType AddAnnotations(
    //         EsiType esiType,
    //         EsiCapnpLocation loc,
    //         IReadOnlyList<Annotation.READER> annotations)
    //     {
    //         return annotations?.Aggregate(esiType, (et, a) => AddAnnotation(et, loc, a)) ?? esiType;
    //     }

    //     public EsiType AddAnnotation (EsiType esiType, EsiCapnpLocation loc, Annotation.READER a) {
    //         if (!ESIAnnotations.Contains( a.Id ))
    //             // No-op if we don't recognize the annotation ID
    //             return esiType;

    //         switch (esiType, (AnnotationIDs) a.Id)
    //         {
    //             // ---
    //             // INLINE annotation
    //             case (EsiReferenceType stRef, AnnotationIDs.INLINE) when stRef.Reference != null:
    //                 return stRef.Reference;
    //             case (EsiReferenceType stRef, AnnotationIDs.INLINE) when stRef.Reference == null:
    //                 C.Log.Error("$Inline found a data type cycle not broken by a reference type ({loc})", loc);
    //                 return esiType;
    //             case (EsiValueType _, AnnotationIDs.INLINE):
    //                 C.Log.Warning("$inline on value types have no effect ({loc})", loc);
    //                 return esiType;
    //             case (_, AnnotationIDs.INLINE):
    //                 C.Log.Error("$Inline on '{type}' not expected ({loc})", esiType.GetType(), loc);
    //                 return esiType;

    //             // ---
    //             // All annotations on refs apply to the thing they reference
    //             case (EsiReferenceType refType, AnnotationIDs.ARRAY): // ARRAY implies $inline
    //                 return AddAnnotation(refType.Reference, loc, a);
    //             case (EsiReferenceType refType, _): // Default case
    //                 return refType.WithReference(AddAnnotation(refType.Reference, loc, a));

    //             // ---
    //             // BITS annotation
    //             case (EsiInt ei, AnnotationIDs.BITS):
    //                 if (ei.Bits < a.Value.Uint64)
    //                 {
    //                     C.Log.Warning(
    //                         "Specified bits ({SpecifiedBits}) is wider than host type holds ({HostBits})! ({loc})",
    //                         a.Value.Uint64,
    //                         ei.Bits,
    //                         loc);
    //                 }
    //                 return new EsiInt(a.Value.Uint64, ei.Signed);
    //             case (EsiContainerType containerType, AnnotationIDs.BITS):
    //                 return containerType.WithInner(AddAnnotation(containerType.Inner, loc, a));
    //             case (_, AnnotationIDs.BITS):
    //                 C.Log.Error("$ESI.bits() can only be applied to integer types! ({loc})", loc);
    //                 return esiType;

    //             // ---
    //             // ARRAY annotation
    //             case (EsiList list, AnnotationIDs.ARRAY):
    //                 return new EsiArray(list.Inner, a.Value.Uint64);
    //             case (EsiStruct st, AnnotationIDs.ARRAY):
    //                 return EsiStructToArray(st, a.Value.Uint64);
    //             case (_, AnnotationIDs.ARRAY):
    //                 C.Log.Error("$Array on '{type}' not valid ({loc})", esiType.GetType(), loc);
    //                 return esiType;

    //             // ---
    //             // C_UNION annotation
    //             case (_, AnnotationIDs.C_UNION):
    //                 C.Log.Error("$cUnion not yet supported");
    //                 return esiType;

    //             // ---
    //             // FIXED_LIST annotation
    //             case (EsiList list, AnnotationIDs.FIXED_LIST):
    //                 return new EsiList(list.Inner, true);
    //             case (_, AnnotationIDs.FIXED_LIST):
    //                 C.Log.Error("$FixedList on '{type}' not valid ({loc})", esiType.GetType(), loc);
    //                 return esiType;

    //             // ---
    //             // FIXED annotation
    //             case (EsiCompound esiCompound, AnnotationIDs.FIXED):
    //                 var cpnpFixedSpec = new FixedPointSpec.READER(a.Value.Struct);
    //                 return EsiCompound.SingletonFor(
    //                     EsiCompound.CompoundType.EsiFixed,
    //                     cpnpFixedSpec.Signed, cpnpFixedSpec.Whole, cpnpFixedSpec.Fraction);
    //             case (_, AnnotationIDs.FIXED):
    //                 C.Log.Error("$Fixed on '{type}' not valid ({loc})", esiType.GetType(), loc);
    //                 return esiType;

    //             // ---
    //             // FLOAT annotation
    //             case (EsiCompound esiCompound, AnnotationIDs.FLOAT):
    //                 var cpnpFloatSpec = new FloatingPointSpec.READER(a.Value.Struct);
    //                 return EsiCompound.SingletonFor(
    //                     EsiCompound.CompoundType.EsiFloat,
    //                     cpnpFloatSpec.Signed, cpnpFloatSpec.Exp, cpnpFloatSpec.Mant);
    //             case (_, AnnotationIDs.FLOAT):
    //                 C.Log.Error("$Float on '{type}' not valid ({loc})", esiType.GetType(), loc);
    //                 return esiType;

    //             // ---
    //             // HWOffset annotation
    //             case (_, AnnotationIDs.HWOFFSET):
    //                 C.Log.Error("$hwoffset not yet supported");
    //                 return esiType;

    //             case (_, _):
    //                 C.Log.Error("Annotation not recognized (annotationID)", a.Id);
    //                 return esiType;
    //         }
    //     }

    //     protected EsiType EsiStructToArray(EsiStruct st, ulong length)
    //     {
    //         if ((ulong)st.Fields.Length != length)
    //         {
    //             C.Log.Error("Groups annotated with $array({n}) need to have a number of elements equal to {n}, not {actual}",
    //                 length, st.Fields.Length);
    //             return st;
    //         }
    //         if (length == 0)
    //         {
    //             // Special case where internal type cannot be determined
    //             return new EsiArray(EsiPrimitive.Void, 0);
    //         }
    //         var inner = st.Fields[0].Type;
    //         return new EsiArray(inner, length);
    //     }
    // }

    // /// <summary>
    // /// Extend ESI's struct reference to add ref counting (for internal,
    // /// cpnp-specific accounting)
    // /// </summary>
    // public class EsiReferenceCapnp : EsiReferenceType
    // {
    //     public long RefCount = 0;

    //     public EsiReferenceCapnp(EsiStruct Reference) : base (Reference)
    //     {    }

    //     public EsiReferenceCapnp(Func<EsiType> Resolver) : base (Resolver)
    //     {    }
    // }

    // /// <summary>
    // /// Delay an error message until type is used... This may or may not be a good idea.
    // /// </summary>
    // public class CapnpEsiErrorType : EsiTypeParent
    // {
    //     public Action A { get; }

    //     public CapnpEsiErrorType(Action A)
    //     {
    //         this.A = A;
    //     }

    //     public override void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
    //     {
    //         A();
    //     }
    // }

    // /// <summary>
    // /// An exception in the conversion process
    // /// </summary>
    // public class EsiCapnpConvertException : Exception
    // {
    //     public EsiCapnpConvertException(string msg) : base (msg) { }
    // }


};

llvm::Error ConvertToESI(
    mlir::MLIRContext* ctxt,
    ParsedSchema& rootSchema,
    vector<mlir::Type>& outputTypes)
{
    CapnpParser cp(ctxt);
    return cp.ConvertTypes(rootSchema.getAllNested(), outputTypes);
}

}
}
