using System.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using Scriban.Runtime;


#nullable enable
namespace Esi
{
    public static class Utils
    {
        private static DirectoryInfo? _Rootdir = null;
        public static DirectoryInfo RootDir
        {
            get
            {
                if (_Rootdir == null)
                {
                    var assem = new FileInfo(Assembly.GetExecutingAssembly().Location);
                    var dir = assem.Directory;
                    while (dir != null)
                    {
                        if (dir.GetDirectories("support").Length > 0)
                        {
                            _Rootdir = dir;
                            break;
                        }
                        dir = dir.Parent;
                    }
                    if (_Rootdir == null)
                    {
                        throw new FileNotFoundException("Could not find root directory for ESI collatoral!");
                    }
                }
                return _Rootdir;
            }
        }

        public static FileInfo FileUnder(this DirectoryInfo me, string filename)
        {
            return new FileInfo(Path.Combine(me.FullName, filename));
        }

        public static FileInfo ResolveResource(string resource)
        {
            return new FileInfo(Path.Combine(RootDir.FullName, resource));
        }

        public static string RenderTemplate(string tmplName, ScriptObject scriptObject)
        {
            var templateContext = new Scriban.TemplateContext {
                MemberRenamer = m => m.Name,
                StrictVariables = true,
                RegexTimeOut = new TimeSpan(0, 0, seconds: 1),
            };
            templateContext.PushGlobal(scriptObject);

            var tmplFile = ResolveResource(Path.Combine("support", tmplName));
            if (!tmplFile.Exists)
                throw new ArgumentException($"Cannot find template '{tmplFile.FullName}'");
            var template = Scriban.Template.Parse (
                File.ReadAllText(tmplFile.FullName),
                sourceFilePath: tmplFile.FullName );
            return template.Render(templateContext);
        }

        public static void RenderTemplate(string tmplName, ScriptObject scriptObject, FileInfo to)
        {
            if (to.Exists)
                to.Delete();
            using (var w = new StreamWriter(to.OpenWrite()))
                w.Write(RenderTemplate(tmplName, scriptObject));
        }

        // ------ More LINQ methods
        public static IList<R> Iterate<T, R>(this IEnumerable<T> collection, Func<T, R> F)
        {
            return collection.Select(F).ToList();
        }

        public static void ForEach<T>(this IEnumerable<T> collection, Action<T> F)
        {
            foreach (var i in collection)
            {
                F(i);
            }
        }

        public static bool ZipAllTrue<T>(this IEnumerable<T> a, IEnumerable<T> b, Func<T, T, bool> F)
        {
            return a.Count() == b.Count() &&
                a.Zip(b, F).All(x => x);
        }

        public static void For(this long Iters, Action<long> action)
        {
            for (long i=0; i<Iters; i++)
                action(i);
        }

        public static void For(this ulong Iters, Action<ulong> action)
        {
            for (ulong i=0; i<Iters; i++)
                action(i);
        }

    }
}