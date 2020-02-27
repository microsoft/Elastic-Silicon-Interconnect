using System.IO;
using System.Reflection;
using NUnit.Framework;
using Esi.Schema;
using System;

namespace  Esi.Core.Tests
{
    public class CapnpTest
    {
        private FileInfo ResolveResource(string resource)
        {
            return Esi.Utils.ResolveResource(Path.Combine("tests", resource));
        }

        public void ReadSchema(string resource)
        {
            EsiCapnpConvert.ConvertTextSchema(ResolveResource(resource));
        }

    }
}