using System.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Capnp;
using CapnpGen;
using CliWrap;

namespace Esi.Schema 
{
    public class EsiCapnpConvert
    {
        public static IReadOnlyList<EsiType> Convert(CodeGeneratorRequest request)
        {
            var convert = new EsiCapnpConvert();
            return convert.GoConvert(request);
        }

        public static IReadOnlyList<EsiType> ConvertFromCGRMessage(Stream stream)
        {
            var frame = Framing.ReadSegments(stream);
            var deserializer = DeserializerState.CreateRoot(frame);
            var reader = CodeGeneratorRequest.READER.create(deserializer);
            var cgr = new CodeGeneratorRequest();
            cgr.Nodes = reader.Nodes.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.Node>(_));
            cgr.RequestedFiles = reader.RequestedFiles.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.CodeGeneratorRequest.RequestedFile>(_));
            cgr.CapnpVersion = CapnpSerializable.Create<CapnpGen.CapnpVersion>(reader.CapnpVersion);
            cgr.applyDefaults();
            return Convert(cgr);
        }

        public static IReadOnlyList<EsiType> ConvertTextSchema(FileInfo file)
        {
            var capnpResult = Cli.Wrap("capnp")
                .SetArguments($"compile -I../../schema/ -o- {file.FullName}")
                .Execute();
            var bytes = Encoding.ASCII.GetBytes(capnpResult.StandardOutput);
            using (var stream = new MemoryStream(bytes) )
            {
                return ConvertFromCGRMessage(stream);
            }
        }


        protected IReadOnlyList<EsiType> GoConvert(CodeGeneratorRequest cgr)
        {
            var ret = cgr.Nodes.Select(ConvertNode).Where(t => t != null);
            return null;
        }

        protected EsiType ConvertNode(Node node)
        {
            switch (node.which)
            {
                case Node.WHICH.Struct:
                    return ConvertStruct(node.Struct);
            }
            return null;
        }

        private EsiType ConvertStruct(Node.@struct s)
        {
            throw new NotImplementedException();
        }
    }
}