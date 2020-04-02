using System.Diagnostics;
using System.Threading;
using Microsoft.Win32.SafeHandles;
using System.Linq;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

#nullable enable
namespace Esi.Schema
{
    /// <summary>
    /// Any class which implements this interface is considered part of the ESI
    /// type system. The various code generators are only required to implement
    /// the core types -- anything in the core library. As such, if other types
    /// are added by other libraries they should be lowered -- if at all possible
    /// -- into these types.
    /// 
    /// The practice which the type classes in this library adhere to is
    /// immutability. This encourages a functional style of programming (or
    /// merely object construction). They also have C# nullable enabled, which
    /// requires that any reference which can be null, declare it. The reason for
    /// this is to maximize compile-time safety. Any classes which extend this
    /// are encouraged to keep the same model, as various library users are free
    /// to assume the immutablity of EsiTypes.
    /// </summary>
    public interface EsiType : EsiObject
    {
        bool StructuralEquals(EsiType that, IDictionary<EsiType, EsiType?>? objMap = null);
    }
    
    /// <summary>
    /// Interface for any type which happens to have a name.
    /// </summary>
    public interface EsiNamedType : EsiType
    {
        string? Name { get; }
    }

    /// <summary>
    /// 'Tag' for types which model an inline or pass-by-value data type. It's
    /// pretty much all of them except for EsiReferenceType.
    /// </summary>
    public interface EsiValueType : EsiType
    {    }

    /// <summary>
    /// Abstraction for things which contain another type (struct fields, lists,
    /// arrays, etc.)
    /// </summary>
    public interface EsiContainerType : EsiType
    {
        EsiType Inner { get; }

        EsiContainerType WithInner(EsiType newInner);
    }

    /// <summary>
    /// Common methods which all EsiTypes could use. The implementation of more
    /// complex methods are in separate files.
    /// </summary>
    public abstract partial class EsiTypeParent : EsiType
    {
        public abstract void GetDescriptionTree(StringBuilder stringBuilder, uint indent);
    }
}
