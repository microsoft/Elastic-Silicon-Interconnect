// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.


#nullable enable
using System.Text;

namespace Esi.Schema
{
    public static class EsiSchemaCommonExts
    {
        public static string? GetFilename(this EsiNamedType namedType)
        {
            return namedType?.Name
                ?.Replace(' ', '_')
                ?.Replace('-', '_');
        }

        public static StringBuilder Indent(this StringBuilder stringBuilder, uint indent)
        {
            for (int i = 0; i < indent; i++)
                stringBuilder.Append("  ");
            return stringBuilder;
        }
    }
}