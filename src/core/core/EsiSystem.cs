using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Esi.Schema;
using System;

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

        /// <summary>
        /// This currently assumes the 'Objects' collections are in the same
        /// order, which is only true in select cases.
        /// </summary>
        public bool StructuralEquals(EsiSystem that)
        {
            if (this.Objects.Count() != that.Objects.Count())
                return false;
            return Objects.ZipAllTrue(that.Objects, (a, b) => 
                (a, b) switch {
                    (EsiType at, EsiType bt) => at.StructuralEquals(bt),
                    (EsiInterface ai, EsiInterface bi) => ai.StructuralEquals(bi),
                    _ when a.GetType() != b.GetType() => false
                });
        }

        public void Traverse(Action<EsiObject> A)
        {
            Objects.ForEach(obj => {obj.Traverse(A);});
        }
    }
}
