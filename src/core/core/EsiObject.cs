using System.Linq;
using System.Collections.Generic;
using System.Text;
using Esi.Schema;
using System;

#nullable enable
namespace Esi
{
    /// <summary>
    /// Most ESI model classes should implement this interface
    /// </summary>
    public interface EsiObject
    {
        void GetDescriptionTree(StringBuilder stringBuilder, uint indent);
        public string GetDescriptionTree(uint indent=0)
        {
            var sb = new StringBuilder();
            GetDescriptionTree(sb, indent);
            return sb.ToString();
        }

        void Traverse(Action<EsiObject> action);
    }
}
