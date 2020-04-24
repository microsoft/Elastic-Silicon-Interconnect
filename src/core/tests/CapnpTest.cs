using System.Collections.Generic;
using System.IO;
using System.Reflection;
using NUnit.Framework;
using Esi;
using Esi.Schema;
using System;
using Esi.Capnp;

namespace Esi.Core.Tests
{
    public class CapnpTest : EsiTest
    {
        public static FileInfo ResolveResource(string resource)
        {
            return Esi.Utils.ResolveResource(Path.Combine("tests", resource));
        }

        public IReadOnlyList<EsiObject> ReadSchema(string resource)
        {
            return EsiCapnpReader.ConvertTextSchema(C, ResolveResource(resource));
        }
    }
}