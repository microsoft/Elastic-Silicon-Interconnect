using CommandLine;

namespace Esi
{
    public class EsiCommonCommandOptions
    {
        [Option('o', "output-dir", Required = false, HelpText = "Output files to this directory")]
        public string OutputDir { get; set; } = ".";
    }
}