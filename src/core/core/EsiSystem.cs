using System.Collections;
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
        public IReadOnlyDictionary<string, EsiNamedType> NamedTypes {
            get
            {
                return Objects
                    .Select(t => t as EsiNamedType)
                    .Where(t => t != null && t.Name != null)
                    .ToDictionary(t => t?.Name!, t => t!);
            }
        }

        public IEnumerable<EsiInterface> Interfaces {
            get
            {
                return Objects
                    .Where(t => t is EsiInterface)
                    .Select(t => (t as EsiInterface)!);
            }
        }

        public EsiSystem (IEnumerable<EsiObject> Objects)
        {
            this.Objects = Objects.ToArray();
        }
    }
}
