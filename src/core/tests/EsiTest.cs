using NUnit.Framework;

namespace Esi.Core.Tests
{
    public class EsiTest
    {
        public EsiContext Context { get; set; }
        public bool ShouldFail { get; set; } = false;

        [SetUp]
        public void BuildContext()
        {
            ShouldFail = false;
            Context = new EsiContext();
        }

        [TearDown]
        public void CloseContext()
        {
            Assert.AreEqual(ShouldFail, Context.Failed);
            Context = null;
        }
    }
}