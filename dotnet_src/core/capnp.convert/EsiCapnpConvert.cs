using System.IO.Pipes;
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
using Esi.Schema;

namespace Esi.Capnp
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

        /// <summary>
        /// ESI context member variables are generally called 'C' so it's easier to log stuff
        /// </summary>
        protected EsiContext C;

        public EsiCapnpConvert(EsiContext ctxt)
        {
            this.C = ctxt;
        }
    }

}