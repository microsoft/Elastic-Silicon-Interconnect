using System.Text;

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
            public EsiType Param { get; }
            public EsiType Return { get; }

            public Method(string Name, EsiType Param, EsiType Return)
            {
                this.Name = Name;
                this.Param = Param;
                this.Return = Return;
            }
        }

        public string Name { get; }
        public Method[] Methods { get; }

        public EsiInterface(string Name, Method[] Methods)
        {
            this.Name = Name;
            this.Methods = Methods;
        }

        public void GetDescriptionTree(StringBuilder stringBuilder, uint indent)
        {
            throw new System.NotImplementedException();
        }
    }
}