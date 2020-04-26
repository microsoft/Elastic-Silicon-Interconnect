using System.Security;
using System.Linq;
using NUnit.Framework;
using Esi.Schema;
using System.IO;
using CliWrap.Exceptions;
using Esi.Capnp;

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
            var types = ReadSchema("stress_tests/stress1.capnp").Objects.ToList();

            Assert.Greater(types.Count, 0);
            var structs = types.Where(t => t is EsiStruct).Select(t => t as EsiStruct);
            Assert.Greater(structs.Count(), 0);
            var examples = structs.Where(s => s.Name == "Example");
            Assert.AreEqual(1, examples.Count());

            var example = examples.First();

            var subExamples = example.Fields.Where(f => f.Name == "subExample");
            Assert.AreEqual(1, subExamples.Count());
            Assert.AreSame(example, (subExamples.First().Type as EsiReferenceType)?.Reference);

            var exampleGroups = example.Fields.Where(f => f.Name == "exampleGroup");
            Assert.AreEqual(1, exampleGroups.Count());
            var exampleGroup = exampleGroups.First();
            Assert.IsInstanceOf(typeof(EsiStruct), exampleGroup.Type);
        }

        [Test]
        public void ReadStress1Compare()
        {
            var types = ReadSchema("stress_tests/stress1.capnp").Objects.ToList();

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
            var types = ReadSchema("stress_tests/stress1.capnp").Objects.ToList();

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
                new EsiStruct.StructField("a", new EsiInt(24, true)),
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
                                new EsiStruct.StructField("street", new EsiReferenceType(new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte)))),
                                new EsiStruct.StructField("city", new EsiList(new EsiPrimitive(EsiPrimitive.PrimitiveType.EsiByte))),
                            }
                        )
                    ),
                    new EsiStruct.StructField(
                        "subExample",
                        new EsiReferenceCapnp(() => ExampleModel)
                    )
                });

        [Test]
        public void ReadStress1Fail()
        {
            ShouldFail = true;

            var types = ReadSchema("stress_tests/stress1_fail.capnp").Objects.ToList();
            Assert.AreEqual(11, C.Errors);
            Assert.AreEqual(0, C.Fatals);
            Assert.True(C.Failed);

            // Assert.Fail("Dummy fail to print out stdout");
        }

        [Test]
        public void ReadStress1FailSyntax()
        {
            ShouldFail = true;

            Assert.Throws<CommandExecutionException>(
                () => ReadSchema("stress_tests/stress1_failsyntax.capnp"));
        }

        [Test]
        public void ReadStress1Interfaces()
        {
            var types = ReadSchema("stress_tests/stress1_synth.capnp").Objects.ToList();

            Assert.Greater(types.Count, 0);
            var structs = types.Where(t => t is EsiStruct).Select(t => t as EsiStruct);
            Assert.Greater(structs.Count(), 0);

            var poly = structs.Where(t => t.Name == "Polynomial3").First();
            var shape = structs.Where(t => t.Name == "Shape").First();

            var interfaces = types.Where(t => t is EsiInterface).Select(t => t as EsiInterface);
            Assert.AreEqual(2, interfaces.Count());

            var polyComps = interfaces.Where(i => i.Name == "Polynomial3Compute");
            Assert.AreEqual(1, polyComps.Count());
            var polyComp = polyComps.First();

            Assert.AreEqual(2, polyComp.Methods.Length);
            var comp = polyComp.Methods.Where(m => m.Name == "compute").First();
            // Context.Log.Information("Expected model: {model}", (ComputeParam as EsiObject).GetDescriptionTree());
            // Context.Log.Information("Actual   model: {model}", comp.Param.GetDescriptionTree());

            Assert.True(ComputeParam1Type.StructuralEquals(comp.Params[0].Type));
            Assert.True(ComputeParam2Type.StructuralEquals(comp.Params[1].Type, includeNames: true));
            Assert.True(comp.Returns[0].Type.StructuralEquals(EsiCompound.SingletonFor(
                Type: EsiCompound.CompoundType.EsiFloat,
                Signed: true,
                Whole: 8,
                Fractional: 23
            )));
        }

        static readonly EsiStruct ComputeParam1Type = Polynomal3Model;
        static readonly EsiType ComputeParam2Type =
            EsiCompound.SingletonFor (
                Type: EsiCompound.CompoundType.EsiFloat,
                Signed: true,
                Whole: 8,
                Fractional: 23
            );
    }
}
