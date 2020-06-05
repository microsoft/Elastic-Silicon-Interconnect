// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using NUnit.Framework;

namespace Esi.Core.Tests
{
    public class EsiTest
    {
        public EsiContext C { get; set; }
        public bool ShouldFail { get; set; } = false;

        [SetUp]
        public void BuildContext()
        {
            ShouldFail = false;
            C = new EsiContext();
        }

        [TearDown]
        public void CloseContext()
        {
            Assert.AreEqual(ShouldFail, C.Failed);
            C = null;
        }
    }
}