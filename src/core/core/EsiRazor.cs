using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using RazorLight;
using RazorLight.Razor;

#nullable enable

namespace Esi
{
    public class RazorEngine
    {
        public readonly static RazorEngine Engine = new RazorEngine();

        protected RazorLightEngine _Engine;

        private RazorEngine()
        {
            _Engine = new RazorLightEngineBuilder()
                .UseProject(new FileSystemRazorProject(Utils.SupportDir.FullName, ".rzr"))
                .UseMemoryCachingProvider()
                .Build();
        }

        public string Render(string tmplFile, object model)
        {
            string? result = null;
            Task.Run(async () => {
                result = await _Engine.CompileRenderAsync(tmplFile, model);
            }).Wait();
            if (result == null)
            {
                throw new TemplateNotFoundException($"Some error occurred rendering file '{tmplFile}'");
            }
            return result;
        }

        public void RenderToFile(string tmplFile, object model, FileInfo output)
        {
            if (output.Exists)
                output.Delete();
            var outContents = Render(tmplFile, model);
            using (var w = new StreamWriter(output.OpenWrite()))
            {
                w.Write(outContents);
            }
        }
    }
}