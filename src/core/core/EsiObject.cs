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

        /// <summary>
        /// Traverse the type tree. Terminate at EsiReferenceType (which contains
        /// a mutable pointer) so as to avoid infinite recursion (cycles).
        /// </summary>
        /// <param name="pre">Call this action before proceeding down. If returns
        /// false, do not continue down</param>
        /// <param name="post">Call this on on the way back up</param>
        void Traverse(Func<EsiObject, bool> pre, Action<EsiObject> post);
    }
}
