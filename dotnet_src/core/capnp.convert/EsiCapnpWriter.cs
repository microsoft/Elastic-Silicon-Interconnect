using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Capnp;
using CapnpGen;
using Esi;
using Esi.Schema;

namespace Esi.Capnp
{
    /// <summary>
    /// Write a capnp schema message from a textual ESI schema
    /// </summary>
    public class EsiCapnpWriter : EsiCapnpConvert
    {
        StreamWriter Writer = null;
        long indent = 0;
        protected void WL(string s="") {
            indent.For(_ => Writer.Write("  "));
            Writer.WriteLine(s);
        }
        protected void W(string s="") {
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
            // Write the header
            // FIXME: Hardcoded ID
            Writer.WriteLine(
@"####################
#### ESI-compatible auto-generated schema
###########

using ESI = import ""/EsiCoreAnnotations.capnp"";
");

            var CapnpID = BitConverter.ToUInt64(sys.SystemHash, 0) | 0x80000000_00000000;
            WL($"@0x{CapnpID:X};");
            WL();

            sys.Objects.ForEach(obj => {
                switch (obj) {
                    case EsiStruct st when !string.IsNullOrWhiteSpace(st.Name):
                        Write(st, 0);
                        break;
                }
            });
        }

        /// <summary>
        /// Write a type as capnp text
        /// </summary>
        /// <param name="type"></param>
        /// <param name="codeOrder"></param>
        protected void Write(EsiStruct st, ushort codeOrder)
        {
            // Use the traverse method to do the recursion
            st.Traverse(
                pre: o => TypeWriteTraversePre(o, ref codeOrder),
                post: TypeWriteTraversePost
            );
            Debug.Assert(indent == 0);
        }

        private bool TypeWriteTraversePre(EsiObject obj, ref ushort codeOrder)
        {
            switch (obj)
            {
                case EsiStruct st:
                    if (indent == 0)
                    {
                        WL($"struct {st.Name} {{");
                    }
                    else
                    {
                        Writer.WriteLine($"group {{");
                    }
                    indent++;
                    break;
                case EsiReferenceType refTy when refTy.Reference is EsiStruct stRef:
                    Writer.WriteLine($"{stRef.Name};");
                    break;
                case EsiStruct.StructField f when f.Type is EsiStruct || f.Type is EsiArray:
                    W($"{f.Name} :");
                    break;
                case EsiStruct.StructField f:
                    W($"{f.Name} @{codeOrder++} :");
                    break;
                case EsiArray array:
                    WriteArray(array, ref codeOrder);
                    break;
                case EsiContainerType containerType when containerType.Inner is EsiNamedType:
                    containerType = containerType.WithInner(new EsiReferenceType(containerType.Inner));
                    Writer.Write($"{GetSimpleKeyword(containerType)} {GetAnnotation(containerType)} $ESI.inline()");
                    Writer.WriteLine(";");
                    break;
                case EsiContainerType containerType when containerType.Inner is EsiReferenceType refType:
                    if (refType is EsiNamedType named)
                    {
                        Writer.Write($"{named.Name} $ESI.inline()");
                        Writer.WriteLine(";");
                    }
                    else
                    {
                        C.Log.Error("References to unnamed types are not supported by CapnProto schemas!");
                    }
                    break;
                case EsiValueType valueType:
                    Writer.Write($"{GetSimpleKeyword(valueType)} {GetAnnotation(valueType)}");
                    Writer.WriteLine(";");
                    break;
            }
            return !(obj is EsiContainerType);
        }

        private void TypeWriteTraversePost(EsiObject obj) {
            switch (obj)
            {
                case EsiStruct st:
                    indent--;
                    WL("}");
                    break;
            }
        }

        private void WriteArray(EsiArray array, ref ushort codeOrder)
        {
            ushort lclCodeOrder = codeOrder;
            Writer.WriteLine($"group $ESI.array({array.Length}) {{");
            indent++;
            for(ulong i = 0; i < array.Length; i++) {
                W($"item{i} @{lclCodeOrder++} :");
                array.Inner.Traverse(
                    pre: o => TypeWriteTraversePre(o, ref lclCodeOrder),
                    post: TypeWriteTraversePost
                );
            };
            indent--;
            WL("}");
            codeOrder = lclCodeOrder;
        }

        private string GetSimpleKeyword(EsiValueType valueType)
        {
            return valueType switch {
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBit => "Bool",
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiBool => "Bool",
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiByte => "UInt8",
                    EsiPrimitive p when p.Type == EsiPrimitive.PrimitiveType.EsiVoid => "Void",

                    EsiInt i when i.Signed && i.Bits <= 8 => $"Int8",
                    EsiInt i when i.Signed && i.Bits <= 16 => $"Int16",
                    EsiInt i when i.Signed && i.Bits <= 32 => $"Int32",
                    EsiInt i when i.Signed && i.Bits <= 64 => $"Int64",
                    EsiInt i when !i.Signed && i.Bits <= 8 => $"UInt8",
                    EsiInt i when !i.Signed && i.Bits <= 16 => $"UInt16",
                    EsiInt i when !i.Signed && i.Bits <= 32 => $"UInt32",
                    EsiInt i when !i.Signed && i.Bits <= 64 => $"UInt64",

                    EsiCompound c when
                        c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                        c.Whole == 8 && c.Fractional == 23 =>
                            "Float32",
                    EsiCompound c when
                        c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                        c.Whole == 11 && c.Fractional == 52 =>
                            "Float64",
                    EsiCompound c when c.Type == EsiCompound.CompoundType.EsiFloat =>
                        $"ESI.FloatingPointValue",
                    EsiCompound c when c.Type == EsiCompound.CompoundType.EsiFixed =>
                        $"ESI.FixedPointValue",

                    EsiArray a when a.Inner is EsiNamedType st => $"List({st.Name})",
                    EsiArray a when a.Inner is EsiValueType vt => $"List({GetSimpleKeyword(vt)})",
            };
        }

        private string GetAnnotation(EsiValueType valueType)
        {
            return valueType switch {
                    EsiPrimitive p => "",
                    EsiInt i => $"$ESI.bits({i.Bits})",
                    EsiStruct st => "$ESI.inline()",

                    EsiCompound c when
                        c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                        c.Whole == 8 && c.Fractional == 23 =>
                            "",
                    EsiCompound c when
                        c.Type == EsiCompound.CompoundType.EsiFloat && c.Signed &&
                        c.Whole == 11 && c.Fractional == 52 =>
                            "",
                    EsiCompound c when c.Type == EsiCompound.CompoundType.EsiFloat =>
                        $"$ESI.float(signed = {(c.Signed?"true":"false")}, exp = {c.Whole}, mant = {c.Fractional})",
                    EsiCompound c when c.Type == EsiCompound.CompoundType.EsiFixed =>
                        $"$ESI.fixed(signed = {(c.Signed?"true":"false")}, whole = {c.Whole}, fraction = {c.Fractional})",

                    EsiArray a when a.Inner is EsiValueType vt => $"{GetAnnotation(vt)} $ESI.array({a.Length})",
            };
        }
    }
}