using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Esi.Schema;
using System;
using System.Diagnostics;
using System.Security.Cryptography;

#nullable enable
namespace Esi
{
    public class EsiSystem
    {
        public byte[]? SystemHash = null;

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
        /// FIXME: This currently assumes the 'Objects' collections are in the same
        /// order, which is only true in select cases.
        /// </summary>
        public bool StructuralEquals(EsiSystem that, bool includeNames = false)
        {
            if (this.NamedTypes.Count() != that.NamedTypes.Count())
                return false;
            return this.NamedTypes.Values.ZipAllTrue(that.NamedTypes.Values, (a, b) => 
                a.StructuralEquals(b, includeNames));
        }

        public void ComputeHash(ulong hash = 0xcc08db7c276d81d8)
        {
            List<byte> bytes = new List<byte>();
            bytes.AddRange(BitConverter.GetBytes(hash));
            Objects.ForEach(o => bytes.AddRange(o.GetDeterministicHash(false)));

            SystemHash = SHA256.Create().ComputeHash(bytes.ToArray());
        }

        public void Traverse(Func<EsiObject, bool> pre, Action<EsiObject> post)
        {
            Objects.ForEach(obj => obj.Traverse(pre, post) );
        }

        public string GetDescriptionTree()
        {
            var sb = new StringBuilder();
            sb.AppendLine("EsiSystem objects [");
            Objects.ForEach(o => {
                sb.Append("  ");
                o.GetDescriptionTree(sb, 1);
                sb.AppendLine();
                sb.AppendLine();
            });
            sb.AppendLine("]");
            return sb.ToString();
        }
    }
}
