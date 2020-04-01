using System.Linq;
using System.Collections.Generic;
using System.Text;
using Esi.Schema;

#nullable enable
namespace Esi
{
    public class EsiSystem
    {
        public IEnumerable<EsiObject> Objects { get; }
        public IReadOnlyDictionary<string, EsiNamedType> NamedTypes { get; }


        public EsiSystem (IEnumerable<EsiObject> Objects)
        {
            this.Objects = Objects.ToArray();
            NamedTypes =
                this.Objects
                    .Select(t => t as EsiNamedType)
                    .Where(t => t != null && t.Name != null)
                    .ToDictionary(t => t?.Name!, t => t!);
        }
    }
}
