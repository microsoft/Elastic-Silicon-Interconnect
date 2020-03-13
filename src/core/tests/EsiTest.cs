using NUnit.Framework;

namespace Esi.Core.Tests
{
    public class EsiTest
    {
        public EsiContext Context { get; set; }

        [SetUp]
        public void BuildContext()
        {
            Context = new EsiContext();
        }

        [TearDown]
        public void CloseContext()
        {
            Assert.False(Context.Failed);
            Context = null;
        }
    }
}