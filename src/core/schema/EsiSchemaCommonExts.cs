
#nullable enable
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
    }
}