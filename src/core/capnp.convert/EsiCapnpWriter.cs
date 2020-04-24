using System;
using System.IO;
using Capnp;
using CapnpGen;

namespace Esi.Capnp
{
    /// <summary>
    /// Write a capnp schema message from an ESI schema
    /// </summary>
    public class EsiCapnpWriter : EsiCapnpConvert
    {
        public EsiCapnpWriter(EsiContext ctxt): base(ctxt)
        {    }

        public void Write(EsiSystem sys, FileInfo into)
        {
            if (into.Exists)
                into.Delete();
            using (var stream = into.OpenWrite())
            {
                Write(sys, stream);
            }
        }

        public void Write(EsiSystem sys, Stream stream)
        {
            var cgr = GetCGR(sys);
            var msg = MessageBuilder.Create();
            var cgrRoot = msg.BuildRoot<CodeGeneratorRequest.WRITER>();
            cgr.serialize(cgrRoot);

            var pump = new FramePump(stream);
            pump.Send(msg.Frame);
        }

        private CodeGeneratorRequest GetCGR(EsiSystem sys)
        {
            throw new NotImplementedException();
        }
    }
}