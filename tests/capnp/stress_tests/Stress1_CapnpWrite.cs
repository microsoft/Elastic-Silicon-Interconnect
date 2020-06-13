// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.Security;
using System.Linq;
using NUnit.Framework;
using Esi.Schema;
using System.IO;
using CliWrap.Exceptions;
using Esi.Capnp;

namespace Esi.Core.Tests 
{
    public class Stress1_CapnpWriteUnitTest : CapnpTest
    {
        [SetUp]
        public void Setup()
        {
        }


        [Test]
        public void ReadWriteCompareStress1Synth()
        {
            var origSys = ReadSchema("stress_tests/stress1_synth.capnp");

            C.Log.Information("Writing cgr");
            var writer = new EsiCapnpWriter(C);
            var file = ResolveResource("stress_tests/schema1_synth.esi.capnp");
            var oldSysFile = ResolveResource("stress_tests/stress1_synth.esimodel.txt");
            File.WriteAllText(oldSysFile.FullName, origSys.GetDescriptionTree());
            writer.Write(origSys, file);

            C.Log.Information("Reading mutated cgr");
            var newSys = ReadSchema("stress_tests/schema1_synth.esi.capnp");
            var newSysFile = ResolveResource("stress_tests/stress1_synth.trans.esimodel.txt");
            File.WriteAllText(newSysFile.FullName, newSys.GetDescriptionTree());
            Assert.True(newSys.StructuralEquals(newSys));
            Assert.True(origSys.StructuralEquals(origSys));
            Assert.True(origSys.StructuralEquals(newSys));
        }
    }
}
