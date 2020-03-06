using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using Esi;
using Esi.Schema;
using System;

namespace  Esi.Core.Tests
{
    public class CapnpTest
    {
        public static FileInfo ResolveResource(string resource)
        {
            return Esi.Utils.ResolveResource(Path.Combine("tests", resource));
        }

        public static IReadOnlyList<EsiType> ReadSchema(string resource)
        {
            return EsiCapnpConvert.ConvertTextSchema(new EsiContext(), ResolveResource(resource));
        }

    }
}