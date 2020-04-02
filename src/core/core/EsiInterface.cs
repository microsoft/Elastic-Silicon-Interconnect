using System.Linq;
using System;
using System.Text;
using System.Collections.Generic;

#nullable enable

namespace Esi.Schema
{
    /// <summary>
    /// An ESI Interface models a collection of channels grouped into 'methods',
    /// each of which may or may not have a parameter or return type. This is
    /// useful for modeling the HW/SW interface. It _may_ also be useful for
    /// on-chip communication but that could just as easily not be the case.
    /// </summary>
    public class EsiInterface : EsiObject
    {
        public struct Method
        {
            public string Name { get; }
            public (string Name, EsiType Type)[] Params { get; }
            public (string Name, EsiType Type)[] Returns { get; }

            public Method(
                string Name,
                IEnumerable<(string name, EsiType)> Params,
                IEnumerable<(string name, EsiType)> Returns)
            {
                this.Name = Name;
                this.Params = Params.ToArray();
                this.Returns = Returns.ToArray();
            }

            public IEnumerable<EsiNamedType> CollectTypes()
            {
                var ret = new List<EsiNamedType>();
                foreach (var p in Params)
                {
                    if (p.Type is EsiNamedType pnt)
                        ret.AddRange(pnt.GetClosestNames());
                }
                foreach (var r in Returns)
                {
                    if (r.Type is EsiNamedType rnt)
                        ret.AddRange(rnt.GetClosestNames());
                }
                return ret;
            }
        }

        public string Name { get; }
        public Method[] Methods { get; }

        public EsiInterface(string Name, IEnumerable<Method> Methods)
        {
            this.Name = Name;
            this.Methods = Methods.ToArray();
        }

        public void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            throw new System.NotImplementedException();
        }

        public IEnumerable<EsiNamedType> CollectTypes()
        {
            return Methods.SelectMany(method => method.CollectTypes());
        }
    }
}