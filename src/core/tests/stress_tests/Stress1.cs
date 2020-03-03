using System.Linq;
using NUnit.Framework;
using Esi.Schema;

namespace Esi.Core.Tests 
{
    public class Stress1UnitTest : CapnpTest
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ReadStress1a()
        {
            var types = ReadSchema("stress_tests/stress1.capnp");
            Assert.Greater(types.Count, 0);
            var structs = types.Where(t => t is EsiStruct).Select(t => t as EsiStruct);
            Assert.Greater(structs.Count(), 0);
            var examples = structs.Where(s => s.Name == "Example");
            Assert.AreEqual(1, examples.Count());

            var example = examples.First();

            var subExamples = example.Fields.Where(f => f.Name == "subExample");
            Assert.AreEqual(1, subExamples.Count());
            Assert.AreSame(example, subExamples.First().Type);

            var exampleGroups = example.Fields.Where(f => f.Name == "exampleGroup");
            Assert.AreEqual(1, exampleGroups.Count());
            var exampleGroup = exampleGroups.First();
            Assert.IsInstanceOf(typeof(EsiStruct), exampleGroup.Type);
        }
    }
}