using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        StreamWriter Writer = null;
        long indent = 0;
        protected void WL(string s) {
            indent.For(_ => Writer.Write("  "));
            Writer.WriteLine(s);
        }
        protected void W(string s) {
            indent.For(_ => Writer.Write("  "));
            Writer.Write(s);
        }

        // UInt64 ctr = 0xc2c4418e9ca66d10;

        public EsiCapnpWriter(EsiContext ctxt): base(ctxt)
        {    }

        public void Write(EsiSystem sys, FileInfo into)
        {
            if (into.Exists)
                into.Delete();

            try
            {
                Writer = new StreamWriter(into.OpenWrite());
                Write(sys);
            } finally {
                Writer.Close();
                Writer = null;
            }
        }

        public void Write(EsiSystem sys)
        {
            Writer.WriteLine(
@"####################
#### ESI-compatible auto-generated schema
###########

using ESI = import ""/EsiCoreAnnotations.capnp"";

@0xb0d0343dd82755a3;
");

            sys.Objects.ForEach(obj => {
                switch (obj) {
                    case EsiNamedType named when !string.IsNullOrWhiteSpace(named.Name):
                        Write(named, 0);
                        break;
                }
            });
        }

        protected void Write(EsiType type, ushort codeOrder)
        {
            type.Traverse (
                pre: obj => {
                    switch (obj)
                    {
                        case EsiStruct st:
                            if (indent == 0) {
                                WL($"struct {st.Name} {{");
                            } else {
                                Writer.WriteLine($"group {{");
                            }
                            indent++;
                            break;
                        case EsiReferenceType refTy when refTy.Reference is EsiStruct stRef:
                            Writer.WriteLine($"{stRef.Name};");
                            break;
                        case EsiStruct.StructField f when f.Type is EsiStruct:
                            W($"{f.Name} :");
                            break;
                        case EsiStruct.StructField f:
                            W($"{f.Name} @{codeOrder++} :");
                            break;
                        case EsiValueType valueType:
                            Writer.Write(GetSimpleKeyword(valueType));
                            Writer.WriteLine(";");
                            break;
                    }
                },
                post: obj => {
                    switch (obj)
                    {
                        case EsiStruct st:
                            indent--;
                            WL("}");
                            break;
                    }
                }
            );
            Debug.Assert(indent == 0);
        }

        private string GetSimpleKeyword(EsiValueType valueType)
        {
            return valueType switch {
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBit => "Bool",
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBool => "Bool",
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiByte => "UInt8",
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiVoid => "Void",

                    EsiInt i when i.Signed && i.Bits <= 8 => $"Int8 $ESI.bits({i.Bits})",
                    EsiInt i when i.Signed && i.Bits <= 16 => $"Int16 $ESI.bits({i.Bits})",
                    EsiInt i when i.Signed && i.Bits <= 32 => $"Int32 $ESI.bits({i.Bits})",
                    EsiInt i when i.Signed && i.Bits <= 64 => $"Int64 $ESI.bits({i.Bits})",
                    EsiInt i when !i.Signed && i.Bits <= 8 => $"UInt8 $ESI.bits({i.Bits})",
                    EsiInt i when !i.Signed && i.Bits <= 16 => $"UInt16 $ESI.bits({i.Bits})",
                    EsiInt i when !i.Signed && i.Bits <= 32 => $"UInt32 $ESI.bits({i.Bits})",
                    EsiInt i when !i.Signed && i.Bits <= 64 => $"UInt64 $ESI.bits({i.Bits})",

                    EsiCompound c when
                        c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                        c.Whole == 8 && c.Fractional == 23 =>
                            "Float32",
                    EsiCompound c when
                        c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                        c.Whole == 11 && c.Fractional == 52 =>
                            "Float64",
                    EsiCompound c when c.Type == EsiCompound.CompoundType.EsiFloat =>
                        $"ESI.FloatingPointValue $ESI.float(signed = {(c.Signed?"true":"false")}, exp = {c.Whole}, mant = {c.Fractional})",
                    EsiCompound c when c.Type == EsiCompound.CompoundType.EsiFixed =>
                        $"ESI.FixedPointValue $ESI.fixed(signed = {(c.Signed?"true":"false")}, whole = {c.Whole}, fraction = {c.Fractional})",
            };
        }
    }
}