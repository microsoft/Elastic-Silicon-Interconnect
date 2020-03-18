using System.Security;
using System.Linq;
using NUnit.Framework;
using Esi.Schema;
using System.IO;
using CliWrap.Exceptions;

namespace Esi.Core.Tests 
{
    public class Stress1UnitTest : CapnpTest
    {
        [SetUp]
        public void Setup()
        {
        }

        /// <summary>
        /// Just test that the parser doesn't crap out entirely
        /// </summary>
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
            Assert.AreSame(example, (subExamples.First().Type as EsiStructReference)?.Struct);

            var exampleGroups = example.Fields.Where(f => f.Name == "exampleGroup");
            Assert.AreEqual(1, exampleGroups.Count());
            var exampleGroup = exampleGroups.First();
            Assert.IsInstanceOf(typeof(EsiStruct), exampleGroup.Type);
        }

        [Test]
        public void ReadStress1Compare()
        {
            var types = ReadSchema("stress_tests/stress1.capnp");

            Assert.Greater(types.Count, 0);
            var structs = types.Where(t => t is EsiStruct).Select(t => t as EsiStruct);
            Assert.Greater(structs.Count(), 0);

            var poly = structs.Where(t => t.Name == "Polynomial3").First();
            Assert.True(Polynomal3Model.StructuralEquals(poly));

            var ex = structs.Where(t => t.Name == "Example").First();
            Assert.True(ExampleModel.StructuralEquals(ex));


        }

        // Test the structural equals code -- various ways it can evaluate false
        [Test]
        public void ReadStress1Incorrect()
        {
            var types = ReadSchema("stress_tests/stress1.capnp");

            Assert.Greater(types.Count, 0);
            var structs = types.Where(t => t is EsiStruct).Select(t => t as EsiStruct);
            Assert.Greater(structs.Count(), 0);

            var poly = structs.Where(t => t.Name == "Polynomial3").First();
            var ex = structs.Where(t => t.Name == "Example").First();

            // Test the structural equals code -- various ways it can evaluate false
            Assert.False(ExampleModel.StructuralEquals(poly));
            Assert.False(Polynomal3Model_Incorrect1.StructuralEquals(poly));
            Assert.False(poly.StructuralEquals(Polynomal3Model_Incorrect1));
            Assert.False(Polynomal3Model_Incorrect2.StructuralEquals(poly));
            Assert.False(poly.StructuralEquals(Polynomal3Model_Incorrect2));
        }

        static readonly EsiStruct Polynomal3Model =
            new EsiStruct("Polynomial3", new EsiStruct.StructField[] {
                new EsiStruct.StructField("a", new EsiInt(24, false)),
                new EsiStruct.StructField("b", new EsiInt(40, false)),
                new EsiStruct.StructField("c", EsiCompound.SingletonFor(
                    EsiCompound.CompoundType.EsiFloat,
                    true,
                    3,
                    10
                )),
            });

        static readonly EsiStruct Polynomal3Model_Incorrect1 =
            new EsiStruct("Polynomial3", new EsiStruct.StructField[] {
                new EsiStruct.StructField("a", new EsiInt(24, false)),
            });
        static readonly EsiStruct Polynomal3Model_Incorrect2 =
            new EsiStruct("Poly3", new EsiStruct.StructField[] { });

        static readonly EsiStruct ExampleModel =
            new EsiStruct("Example",
                new EsiStruct.StructField[] {
                    new EsiStruct.StructField(
                        "poly",
                        Polynomal3Model
                    ),
                    new EsiStruct.StructField(
                        "exampleGroup",
                        new EsiStruct (
                            null,
                            new EsiStruct.StructField[] {
                                new EsiStruct.StructField("houseNumber", new EsiInt(32, false)),
                                new EsiStruct.StructField("street", new EsiListReference(new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte)))),
                                new EsiStruct.StructField("city", new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte))),
                            }
                        )
                    ),
                    new EsiStruct.StructField(
                        "subExample",
                        new EsiStructReferenceCapnp(() => ExampleModel)
                    )
                });

        [Test]
        public void ReadStress1Fail()
        {
            var types = ReadSchema("stress_tests/stress1_fail.capnp");
            Assert.AreEqual(10, Context.Errors);
            Assert.AreEqual(0, Context.Fatals);
            Assert.True(Context.Failed);
            Context.ClearCounts();

            // Assert.Fail("Dummy fail to print out stdout");
        }

        [Test]
        public void ReadStress1FailSyntax()
        {
            Assert.Throws<CommandExecutionException>(
                () => ReadSchema("stress_tests/stress1_failsyntax.capnp"));
            Context.ClearCounts();
        }
    }
}