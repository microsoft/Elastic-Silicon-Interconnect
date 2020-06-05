// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.IO;
using Esi.Schema;
using Scriban.Runtime;


namespace Esi.SVCodeGen
{
    public class SVRender : ScriptObject
    {
        public static string GetSVIdentifier(EsiNamedType namedType) => namedType.GetSVIdentifier();
        public static string GetSVIdentifierIFace(EsiInterface iface) => iface.GetSVIdentifier();
        public static string GetSVHeaderName(EsiType type) => type.GetSVHeaderName();
        public static string GetSVCompoundModuleName(EsiCompound c) => c.GetSVCompoundModuleName();
    }

    public static class SVUtils
    {
        public static string RenderTemplate(string tmplName, ScriptObject scriptObject)
        {
            scriptObject.Import(typeof(SVRender), renamer: m => m.Name);
            return Utils.RenderTemplate(tmplName, scriptObject);
        }

        public static void RenderTemplate(string tmplName, ScriptObject scriptObject, FileInfo to)
        {
            if (to.Exists)
                to.Delete();
            using (var w = new StreamWriter(to.OpenWrite()))
                w.Write(RenderTemplate(tmplName, scriptObject));
        }
    }
}