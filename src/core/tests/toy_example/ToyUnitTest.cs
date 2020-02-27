using System.Reflection;
using NUnit.Framework;

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
            ReadSchema("toy_example/toy.capnp");
        }
    }
}