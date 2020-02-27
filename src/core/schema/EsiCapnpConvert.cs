using System.Diagnostics;
using System.Threading.Tasks;
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
            using (var memstream = new MemoryStream() )
            {
                var capnpCmd = Cli.Wrap("capnp")
                    .WithArguments($"compile -I{Path.Join(Esi.Utils.RootDir.FullName, "schema")} -o- {file.FullName}")
                    .WithStandardOutputPipe(PipeTarget.ToStream(memstream))
                    .WithValidation(CommandResultValidation.ZeroExitCode);

                Task.Run(async () => await capnpCmd.ExecuteAsync()).Wait();
                Debug.Assert(memstream.Length > 0);

                memstream.Seek(0, SeekOrigin.Begin);
                return ConvertFromCGRMessage(memstream);
            }
        }


        protected IReadOnlyList<EsiType> GoConvert(CodeGeneratorRequest cgr)
        {
            var ret = cgr.Nodes.Select(ConvertNode).Where(t => t != null).ToList();
            return ret;
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