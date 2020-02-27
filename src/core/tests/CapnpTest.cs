using System.IO;
using System.Reflection;
using NUnit.Framework;
using Esi.Schema;

namespace  Esi.Core.Tests
{
    public class CapnpTest
    {
        public FileInfo ResolveResource(string resource)
        {
            var assem = new FileInfo(Assembly.GetExecutingAssembly().Location);
            var dir = assem.Directory;
            while (dir != null)
            {
                if (dir.GetDirectories("tests").Length > 0)
                {
                    return new FileInfo(Path.Combine(dir.FullName, "tests", resource));
                }
                dir = dir.Parent;
            }
            Assert.Fail("Could not find 'tests' directory for tests collatoral!");
            return null;
        }

        public void ReadSchema(string resource)
        {
            EsiCapnpConvert.ConvertTextSchema(ResolveResource(resource));
        }
    }
}