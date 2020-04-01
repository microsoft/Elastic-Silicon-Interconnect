using System.Text;

#nullable enable

namespace Esi.Schema
{
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