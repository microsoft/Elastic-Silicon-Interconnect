using System.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;


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
                        if (dir.GetDirectories("schema").Length > 0)
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
    }
}