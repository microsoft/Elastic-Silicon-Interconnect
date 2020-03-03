using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using Esi.Schema;
using System;

namespace  Esi.Core.Tests
{
    public class CapnpTest
    {
        private static FileInfo ResolveResource(string resource)
        {
            return Esi.Utils.ResolveResource(Path.Combine("tests", resource));
        }

        public static IReadOnlyList<EsiType> ReadSchema(string resource)
        {
            return EsiCapnpConvert.ConvertTextSchema(ResolveResource(resource));
        }

    }
}