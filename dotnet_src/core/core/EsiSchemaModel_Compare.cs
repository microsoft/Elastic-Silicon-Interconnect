// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using System.Linq;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Collections;
using System.Security.Cryptography;
using System.Text;

#nullable enable
namespace Esi.Schema
{
    public abstract partial class EsiTypeParent : EsiType
    {
        /// <summary>
        /// Reflection-based equality comparison which handles cycles
        /// </summary>
        /// <param name="that">Object against which to compare</param>
        /// <param name="includeNames">Match names as well?</param>
        /// <param name="objMap">Cache results thus far to be able to handle cycles</param>
        /// <returns></returns>
        public virtual bool StructuralEquals(
            EsiType that,
            bool includeNames = false,
            IDictionary<EsiType, EsiType?>? objMap = null)
        {
            if (that == null)
                return false;
            objMap = objMap ?? new Dictionary<EsiType, EsiType?>();

            Type thisType = this.GetType();
            Type thatType = that.GetType();

            if (thisType != thatType)
                return false;

            foreach (var prop in thisType.GetProperties())
            {
                var thisValue = prop.GetValue(this);
                var thatValue = prop.GetValue(that);

                if (thisValue is EsiTypeParent thisFieldEsi &&
                    thatValue is EsiTypeParent thatFieldEsi)
                {
                    if (objMap.TryGetValue(thisFieldEsi, out var expectedThatField))
                    {
                        if (thatValue != expectedThatField)
                            return false;
                    }
                    else
                    {
                        if (thisFieldEsi == null)
                        {
                            if (thatFieldEsi != null)
                                return false;
                        }
                        else
                        {
                            objMap[thisFieldEsi] = thatFieldEsi;
                            if (!thisFieldEsi.StructuralEquals(thatFieldEsi, includeNames, objMap))
                                return false;
                        }
                    }
                }
                else if (thisValue is IEnumerable thisCollection &&
                         thatValue is IEnumerable thatCollection)
                {
                    if ((thisCollection == null && thatCollection != null) ||
                        (thisCollection != null && thatCollection == null))
                    {
                        return false;
                    }
                    if (thatCollection != null && thisCollection != null)
                    {
                        IEnumerator e1 = thisCollection.GetEnumerator();
                        IEnumerator e2 = thatCollection.GetEnumerator();
                        while (e1.MoveNext())
                        {
                            if (!(e2.MoveNext() && StructuralEquals(e1.Current, e2.Current, includeNames, objMap)))
                                return false;
                        }
                        if (e2.MoveNext())
                            return false;
                    }
                }
                else if (this is EsiNamedType &&
                         that is EsiNamedType &&
                         prop.Name == "Name" &&
                         !includeNames)
                {
                    // Don't include names
                }
                else
                {
                    if ((thisValue != null || thatValue != null) &&
                         thisValue?.Equals(thatValue) != true)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        protected static bool StructuralEquals(object thisValue, object thatValue, bool includeNames, IDictionary<EsiType, EsiType?> objMap)
        {
            if (thisValue is EsiType e1 &&
                thatValue is EsiType e2)
            {
                return e1.StructuralEquals(e2, includeNames, objMap);
            }
            if ((thisValue != null || thatValue != null) &&
                 thisValue?.Equals(thatValue) != true)
            {
                return false;
            }
            return true;
        }

        public byte[] GetDeterministicHash(bool includeNames)
        {
            var seen = new HashSet<EsiType>();
            var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA512);
            AppendHashData(hash, includeNames, seen);
            return hash.GetHashAndReset();
        }

        public virtual void AppendHashData (
            IncrementalHash hash,
            bool includeNames,
            ISet<EsiType> seen)
        {
            seen.Add(this);

            Type thisType = this.GetType();
            foreach (var prop in thisType.GetProperties())
            {
                var thisValue = prop.GetValue(this);
                if (!thisType.IsPublic)
                {
                    // Only look at public fields
                }
                else if (thisValue is IEnumerable thisCollection)
                {
                    foreach (var o in thisCollection)
                    {
                        hash.AppendData(H(o));
                    }
                }
                else if (this is EsiNamedType &&
                         prop.Name == "Name" &&
                         !includeNames)
                {
                    // Don't include names
                }
                else
                {
                    hash.AppendData(H(thisValue));
                }
            }

            byte[] H(object o)
            {
                if (o == null)
                    return new byte[]{};
                var ty = o.GetType();
                if (ty.IsPrimitive)
                    return BitConverter.GetBytes((dynamic)o);
                switch (o) {
                    case string s:
                        return Encoding.UTF8.GetBytes(s);
                    case EsiTypeParent esiType:
                        if (!seen.Contains(esiType))
                        esiType.AppendHashData(hash, includeNames, seen);
                        break;
                    case EsiObject esiObject:
                        return esiObject.GetDeterministicHash();
                }
                return new byte[]{};
            }
        }
    }
}
