// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef ESI_OPS
#define ESI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "EsiDialect.td"

// Base class for other typedefs. Provides dialact-specific defaults
class Esi_Type<string name> : TypeDef<Esi_Dialect, name> { }


// def SimpleTypeA : Esi_Type<"SimpleA"> {
//     let mnemonic = "smpla";
// }

// // A more complex parameterized type
// def CompoundTypeA : Esi_Type<"CompoundA"> {
//     // Override the default mnemonic
//     let mnemonic = "cmpnd_a";

//     // What types do we contain?
//     let members = (
//         ins
//         // A standard c++ int
//         "int":$widthOfSomething,
//         // The simple type defined above
//         "SimpleAType": $exampleTdType,
//         // Some C++ type
//         // "CompoundAType::SomeCppStruct": $exampleCppType
//         "float": $f,
//         "double": $d,
//         ArrayRefMember<"int">: $arrayOfInts,
//         ArrayRefMember<"Type">: $arrayOfTypes,
//         "StringRef": $simpleString,
//         ArrayRefMember<"StringRef">: $arrayOfStrings
//     );

//     // let genVerifyInvariantsDecl = 1;

//     let extraDecls = [{
//         struct SomeCppStruct {};
//     }];
// }

  def FixedPoint : Esi_Type<"FixedPoint"> {
      let mnemonic = "fixed";

      let members = (
          ins
          "bool": $isSigned,
          "size_t": $wholeBits,
          "size_t": $fractionalBits
      );
  }

  def FloatingPoint : Esi_Type<"FloatingPoint"> {
      let mnemonic = "float";

      let members = (
          ins
          "bool": $isSigned,
          "size_t": $expontentBits,
          "size_t": $mantissaBits
      );
  }

  def ListType : Esi_Type<"List"> {
    let mnemonic = "list";

    let members = (
      ins
      "Type": $inner
    );
  }

  class FieldInfo_Type<string name> : Esi_Type<name> {
    let members = (
      ins
      ArrayRefOfSelfAllocationMember<"FieldInfo">: $fields
    );

    let printer = [{
      printer << "struct" << "<";
      for (size_t i=0; i<getImpl()->fields.size(); i++) {
        const auto& field = getImpl()->fields[i];
        printer << "{" << field.name << "," << field.type << "}";
        if (i < getImpl()->fields.size() - 1)
          printer << ",";
      }
      printer << ">";
    }];

    let parser = [{
      SmallVector<FieldInfo, 4> members;
      if (parser.parseLess()) return Type();
      while (mlir::succeeded(parser.parseOptionalLBrace())) {
        StringRef name;
        if (parser.parseKeyword(&name)) return Type();
        if (parser.parseComma()) return Type();
        Type type;
        if (parser.parseType(type)) return Type();
        if (parser.parseRBrace()) return Type();
        members.push_back(FieldInfo {.name = name, .type=type});
        if (parser.parseOptionalComma()) break;
      }
      if (parser.parseGreater()) return Type();
      return get(ctxt, members);
    }];
  }

  def StructType : FieldInfo_Type<"Struct"> {
    let mnemonic = "struct";
  }

  def UnionType : FieldInfo_Type<"Union"> {
    let mnemonic = "union";
  }

  def EnumType : Esi_Type<"Enum"> {
    let mnemonic = "Enum";

    let members = (
      ins
      ArrayRefMember<"StringRef">: $fields
    );
  }

  def Esi_Type : Type<CPred<"$_self.getDialect().getTypeID() == TypeID::get<EsiDialect>()">, "ESI type">;
#endif
