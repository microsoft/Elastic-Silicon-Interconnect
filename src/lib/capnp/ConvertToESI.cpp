// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "Dialects/Esi/EsiTypes.hpp"
#include "capnp/CapnpConvert.hpp"
#include <capnp/schema-parser.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/IR/StandardTypes.h>

#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <printf.h>
#include <tuple>
#include <vector>

using namespace mlir::esi;
using namespace capnp;
using namespace capnp::schema;

namespace esi {
namespace capnp {

class CapnpParser {

  /// <summary>
  /// Track various data and metadata about capnp types
  /// </summary>
  struct EsiCapnpLocation {
  private:
    EsiCapnpLocation &operator=(const EsiCapnpLocation &) = delete;
    EsiCapnpLocation(const EsiCapnpLocation &) = default;

  public:
    StructSchema node;
    mlir::Type type;
    std::string nodeName;
    std::string file;
    std::list<std::string> path;
    bool beingParsed;

    EsiCapnpLocation(StructSchema node, std::string nodeName)
        : node(node), nodeName(nodeName), beingParsed(false) {}

    EsiCapnpLocation AppendField(std::string field) {
      EsiCapnpLocation ret = *this;
      ret.path.push_back(field);
      return ret;
    }

    std::string ToString() {
      std::string fileStruct;
      auto displayName = node.getProto().getDisplayName();
      if (!std::all_of(displayName.begin(), displayName.end(), iswspace))
        fileStruct = displayName;
      else
        fileStruct = llvm::formatv("{0}:{1}", file, nodeName);

      if (path.size() > 0)
        return llvm::formatv("{0}/{1}", fileStruct, llvm::join(path, ","));
      else
        return fileStruct;
    }
  };

  mlir::MLIRContext *ctxt;

  // Resolve the capnp node location from the capnp node id. Used to avoid
  // re-converting types and detecting self-recusion.
  std::map<uint64_t, EsiCapnpLocation> IDtoLoc;

  /// Look up the location instance. Create if doesn't exist
  EsiCapnpLocation &GetLocation(StructSchema node, std::string name = "") {
    auto id = node.getProto().getId();
    auto locIter = IDtoLoc.find(id);
    if (locIter == IDtoLoc.end()) {
      auto nodeName =
          (name.length() == 0) ? node.getProto().getDisplayName().cStr() : name;
      IDtoLoc.emplace(std::piecewise_construct, std::forward_as_tuple(id),
                      std::forward_as_tuple(node, nodeName));
    }
    return IDtoLoc.find(id)->second;
  }

public:
  CapnpParser(mlir::MLIRContext *ctxt) : ctxt(ctxt) {}

  llvm::Error ConvertTypes(::capnp::ParsedSchema schema,
                           std::map<std::string, mlir::Type> &outputTypes) {
    for (auto node : schema.getAllNested()) {
      switch (node.getProto().which()) {
      case ::capnp::schema::Node::Which::STRUCT:
        if (mlir::failed(ConvertStruct(GetLocation(node.asStruct())))) {
          llvm::errs() << llvm::formatv(
              "Failed to convert '{0}' to EsiStruct\n",
              node.getProto().getDisplayName());
        }
        break;
      default:
        llvm::errs() << llvm::formatv("Unconvertable type (node '{0}')\n",
                                      node.getProto().getDisplayName());
        break;
      }
    }

    for (auto &locIdPair : IDtoLoc) {
      outputTypes[locIdPair.second.nodeName] = locIdPair.second.type;
    }

    return llvm::Error::success();
  }

  /// <summary>
  /// Convert a top-level node to an EsiType, lazily
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  mlir::LogicalResult ConvertStruct(EsiCapnpLocation &loc) {
    if (loc.type != mlir::Type())
      return mlir::success();
    if (loc.beingParsed) {
      llvm::errs()
          << "Recursive (and co-recursive) types not supported yet! Found at "
          << loc.ToString() << ".\n";
      return mlir::failure();
    }

    loc.beingParsed = true;
    const auto &fields = loc.node.getFields();
    const size_t num = fields.size();

    std::vector<FieldInfo> esiFields(num, FieldInfo("", nullptr));
    for (size_t i = 0; i < num; i++) {
      auto rc = ConvertField(loc, fields[i], esiFields[i]);
    }
    loc.type = StructType::get(ctxt, esiFields);
    loc.beingParsed = false;
    return mlir::success();
  }

  //     private EsiInterface ConvertInterface(ulong id)
  //     {
  //         var loc = IDtoNames[id];
  //         if (!loc.Node.HasValue)
  //             throw new EsiCapnpConvertException($"Could not find node for
  //             interface {loc}");
  //         var iface = loc.Node.Value.Interface;
  //         var superClasses = iface.Superclasses.Select(s =>
  //         ConvertInterface(s.Id)).ToArray(); return new EsiInterface(
  //             Name: loc.NodeName,
  //             Methods: iface.Methods.Select(ConvertMethod).ToArray()
  //         );
  //     }

  //     private EsiInterface.Method ConvertMethod(Method.READER method)
  //     {
  //         IEnumerable<(string Name, EsiType type)> MethodConvert(EsiType t)
  //         {
  //             if (!(t is EsiReferenceType refType))
  //                 throw new EsiCapnpConvertException($"Internal error:
  //                 expected reference, got {t.GetType()}");

  //             t = refType.Reference;
  //             if (!(t is EsiStruct st))
  //                 throw new EsiCapnpConvertException($"Internal error:
  //                 expected struct reference, got {t.GetType()}*");

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
  //     /// Return a function which returns an EsiStruct which has been
  //     converted from a
  //     /// CapNProto struct.
  //     /// </summary>
  //     private EsiType ConvertStructCached(ulong id)
  //     {
  //         var loc = IDtoNames[id];
  //         if (!IDtoType.TryGetValue(loc.Id, out var esiType))
  //         {
  //             // First, create a struct reference and populate the cache with
  //             it.
  //             //  This signals that we are descending this struct already, in
  //             the case of a cycle var stRef = new
  //             EsiReferenceCapnp((EsiStruct)null); esiType = stRef;
  //             IDtoType[loc.Id] = esiType;

  //             var canpnStruct = loc.Node.Value.Struct;
  //             if (canpnStruct.DiscriminantCount == 0) // This capnp struct is
  //             not a union
  //             {
  //                 var esiStruct = new EsiStruct(
  //                     Name: loc.NodeName,
  //                     Fields: canpnStruct.Fields.Iterate(f =>
  //                     ConvertField(loc, f))
  //                 );
  //                 if (canpnStruct.IsGroup) // This capnp "struct" is actually
  //                 a group, which is equivalent to an EsiStruct
  //                 {
  //                     esiType = esiStruct; // Set the return to the raw
  //                     struct if (stRef.RefCount > 0) // Check to see that
  //                     nobody got the tentative reference while we were workin
  //                         C.Log.Fatal("Found a cycle involving groups. This
  //                         shouldn't occur! ({loc})", loc);
  //                     IDtoType[loc.Id] = esiStruct; // And remove the
  //                     reference to it
  //                 }
  //                 else // This capnp "struct" is actually a capnp struct,
  //                 which is equivalent to an EsiStructReference
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
  //                 throw new EsiCapnpConvertException($"GetNamedNode failed to
  //                 find named node {structId}");
  //         }
  //     }

  //     private EsiType GetNamedType(UInt64 structId)
  //     {
  //         var esiObj = GetNamedNode(structId);
  //         if (esiObj is EsiType type)
  //             return type;
  //         C.Log.Error("Unsupported use as data type: {type}",
  //         esiObj?.GetType()); return null;
  //     }

  /// <summary>
  /// Convert a struct field which can be either an actual member, "slot", or a
  /// group.
  /// </summary>
  mlir::LogicalResult ConvertField(EsiCapnpLocation &loc,
                                   const StructSchema::Field &field,
                                   FieldInfo &fi) {
    llvm::StringRef name = field.getProto().hasName()
                               ? field.getProto().getName().cStr()
                               : "anonymous";
    auto fieldLoc = loc.AppendField(std::string(name));
    auto mlirType = ConvertType(fieldLoc, field.getType());
    if (mlirType == mlir::Type())
      return mlir::failure();
    fi = FieldInfo(name, mlirType);
    return mlir::success();
  }

  /// <summary>
  /// Entry point for recursion. Should handle any embeddable type and its
  /// annotations.
  /// </summary>
  /// <param name="loc"></param>
  /// <param name="type"></param>
  /// <param name="annotations"></param>
  /// <returns></returns>
  mlir::Type ConvertType(EsiCapnpLocation &loc, ::capnp::Type type) {
    mlir::Type mlirType;

    switch (type.which()) {
    case ::capnp::schema::Type::Which::VOID:
      mlirType = mlir::NoneType::get(ctxt);
      break;
    case ::capnp::schema::Type::Which::BOOL:
      mlirType = mlir::IntegerType::get(
          1, mlir::IntegerType::SignednessSemantics::Signless, ctxt);
      break;
    case ::capnp::schema::Type::Which::INT8:
      mlirType = mlir::IntegerType::get(
          8, mlir::IntegerType::SignednessSemantics::Signed, ctxt);
      break;
    case ::capnp::schema::Type::Which::INT16:
      mlirType = mlir::IntegerType::get(
          16, mlir::IntegerType::SignednessSemantics::Signed, ctxt);
      break;
    case ::capnp::schema::Type::Which::INT32:
      mlirType = mlir::IntegerType::get(
          32, mlir::IntegerType::SignednessSemantics::Signed, ctxt);
      break;
    case ::capnp::schema::Type::Which::INT64:
      mlirType = mlir::IntegerType::get(
          64, mlir::IntegerType::SignednessSemantics::Signed, ctxt);
      break;
    case ::capnp::schema::Type::Which::UINT8:
      mlirType = mlir::IntegerType::get(
          8, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt);
      break;
    case ::capnp::schema::Type::Which::UINT16:
      mlirType = mlir::IntegerType::get(
          16, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt);
      break;
    case ::capnp::schema::Type::Which::UINT32:
      mlirType = mlir::IntegerType::get(
          32, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt);
      break;
    case ::capnp::schema::Type::Which::UINT64:
      mlirType = mlir::IntegerType::get(
          64, mlir::IntegerType::SignednessSemantics::Unsigned, ctxt);
      break;
    case ::capnp::schema::Type::Which::FLOAT32:
      mlirType = mlir::esi::FloatingPointType::get(ctxt, true, 8, 23);
      break;
    case ::capnp::schema::Type::Which::FLOAT64:
      mlirType = mlir::esi::FloatingPointType::get(ctxt, true, 11, 52);
      break;
    case ::capnp::schema::Type::Which::TEXT:
      mlirType = mlir::esi::MessagePointerType::get(
          ctxt, mlir::esi::StringType::get(ctxt));
      break;
    case ::capnp::schema::Type::Which::DATA:
      mlirType = mlir::esi::MessagePointerType::get(
          ctxt,
          mlir::esi::ListType::get(
              ctxt,
              mlir::IntegerType::get(
                  8, mlir::IntegerType::SignednessSemantics::Signless, ctxt)));
      break;
    case ::capnp::schema::Type::Which::LIST: {
      auto capnpInner = type.asList().getElementType();
      mlir::Type inner = ConvertType(loc, capnpInner);
      mlirType = mlir::esi::MessagePointerType::get(
          ctxt, mlir::esi::ListType::get(ctxt, inner));
    } break;
    // case ::capnp::schema::Type::Which::ENUM:
    //     mlirType = GetNamedType(type.Enum.TypeId),
    //     break;
    case ::capnp::schema::Type::Which::STRUCT: {
      auto &loc = GetLocation(type.asStruct());
      if (mlir::failed(ConvertStruct(loc))) {
        return mlir::Type();
      }
      mlirType = mlir::esi::MessagePointerType::get(ctxt, loc.type);
    } break;

    //     mi.mlirType = type.Struct.TypeId switch {
    //         break;
    //     // ---
    //     // "Special", known structs
    //     (ulong)AnnotationIDs.FIXED_POINT_VALUE =>
    //         EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFixed, true,
    //         63, 64),
    //     (ulong)AnnotationIDs.FLOATING_POINT_VALUE =>
    //         EsiCompound.SingletonFor(EsiCompound.CompoundType.EsiFloat, true,
    //         63, 64),

    //     // ---
    //     // User-specified structs
    //     _ => GetNamedType(type.Struct.TypeId)
    // },

    // CapnpGen.Type.WHICH.Interface => new CapnpEsiErrorType( () =>
    // C.Log.Error("ESI does not support the Interface type ({loc})", loc) ),
    // CapnpGen.Type.WHICH.AnyPointer => new CapnpEsiErrorType( () =>
    // C.Log.Error("ESI does not support the AnyPointer type ({loc})", loc) ),

    // _ => throw new
    // NotImplementedException($"ConvertType({Enum.GetName(typeof(CapnpGen.Type.WHICH),
    // type.which)}) not implemented ({loc})")
    default:
      llvm::errs() << llvm::formatv(
          "Capnp type number {0} not supported (at {1})\n", type.which(),
          loc.ToString());
      return mlir::Type();
    }

    return mlirType;
  }
}; // namespace capnp

llvm::Error ConvertToESI(mlir::MLIRContext *ctxt, ParsedSchema &rootSchema,
                         std::map<std::string, mlir::Type> &outputTypes) {
  CapnpParser cp(ctxt);
  return cp.ConvertTypes(rootSchema, outputTypes);
  // return llvm::Error::success();
}

} // namespace capnp
} // namespace esi
