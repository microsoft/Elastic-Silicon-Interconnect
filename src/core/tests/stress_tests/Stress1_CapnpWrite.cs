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
            var origSys = new EsiSystem(ReadSchema("stress_tests/stress1_synth.capnp"));
            var writer = new EsiCapnpWriter(C);
            var file = new FileInfo("schema1_synth.esi.capnp");
            writer.Write(origSys, file);

            var newSys = EsiCapnpReader.ReadFromCGR(C, file);
            Assert.True(origSys.StructuralEquals(newSys));
        }


    }
}
