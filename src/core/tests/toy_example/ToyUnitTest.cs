using System.Linq;
using NUnit.Framework;
using Esi.Schema;

namespace Esi.Core.Tests 
{
    public class ToyUnitTest : CapnpTest
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ReadSchemaTest()
        {
            var types = ReadSchema("toy_example/toy.capnp");
            Assert.Greater(types.Count, 0);
            var structs = types.Where(t => t is EsiStruct).Select(t => t as EsiStruct);
            Assert.Greater(structs.Count(), 0);
            var poly3s = structs.Where(s => s.Name == "Polynomial3");
            Assert.AreEqual(1, poly3s.Count());

            var poly3 = poly3s.First();
            Assert.AreEqual(1, poly3.Fields.Where(f => f.Name == "a").Count());
            Assert.IsInstanceOf(typeof(EsiStruct), poly3);

            var poly3A = poly3.FieldLookup["a"];
            Assert.True((new EsiInt(24, false)).StructuralEquals(poly3A.Type));
        }
    }
}