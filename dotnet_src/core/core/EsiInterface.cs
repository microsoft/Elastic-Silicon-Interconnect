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

            public bool StructuralEquals(Method that, bool includeNames)
            {
                bool ParamReturnEqual((string Name, EsiType Type) a, (string Name, EsiType Type) b)
                {
                    return a.Name == b.Name && a.Type.StructuralEquals(b.Type, includeNames);
                }

                return this.Name == that.Name &&
                    this.Params.ZipAllTrue(that.Params, ParamReturnEqual) &&
                    this.Returns.ZipAllTrue(that.Returns, ParamReturnEqual);
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
            // TODO
            // throw new System.NotImplementedException();
        }

        public bool StructuralEquals(EsiInterface that, bool includeNames = false)
        {
            return this.Name == that.Name &&
                this.Methods.ZipAllTrue(that.Methods, (a, b) => a.StructuralEquals(b, includeNames));
        }

        public void Traverse(Func<EsiObject, bool> pre, Action<EsiObject> post)
        {
            pre(this);
            post(this);
        }

        public byte[] GetDeterministicHash(bool includeNames = false)
        {
            // FIXME: obviously wrong
            return new byte[] {};
        }
    }
}