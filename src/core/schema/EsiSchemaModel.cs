namespace schema
{
    public interface EsiType { }

    public class EsiPrimitive : EsiType
    {
        public enum Type {
            EsiBool,
            EsiByte,
            EsiBit,
            EsiUInt,
            EsiInt
        }

        public Type Type { get; }
        public int Bits { get; }
    }

    public class EsiCompound : EsiType
    {
        public enum Type
        {
            EsiFixed,
            EsiFloat,
            EsiUFixed,
            EsiUFloat
        }
    }
}
