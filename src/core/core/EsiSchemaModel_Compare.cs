using System.Linq;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Collections;

#nullable enable
namespace Esi.Schema
{
    public abstract partial class EsiTypeParent : EsiType
    {
        /// <summary>
        /// Reflection-based equality comparison which handles cycles
        /// </summary>
        public virtual bool StructuralEquals(
            EsiType that,
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
                            if (!thisFieldEsi.StructuralEquals(thatFieldEsi, objMap))
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
                            if (!(e2.MoveNext() && StructuralEquals(e1.Current, e2.Current, objMap)))
                                return false;
                        }
                        if (e2.MoveNext())
                            return false;
                    }
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

        protected static bool StructuralEquals(object thisValue, object thatValue, IDictionary<EsiType, EsiType?> objMap)
        {
            if (thisValue is EsiType e1 &&
                thatValue is EsiType e2)
            {
                return e1.StructuralEquals(e2, objMap);
            }
            if ((thisValue != null || thatValue != null) &&
                 thisValue?.Equals(thatValue) != true)
            {
                return false;
            }
            return true;
        }
    }
}
