using Serilog;
using Serilog.Core;

namespace Esi
{
    public class EsiContext
    {
        public Logger Log { get; protected set; }

        public EsiContext(LoggerConfiguration? loggerConfiguration = null)
        {
            if (loggerConfiguration == null)
            {
                loggerConfiguration = new LoggerConfiguration()
                    .MinimumLevel.Debug()
                    .WriteTo.Console();
            }
            Log = loggerConfiguration.CreateLogger();
        }
    }
}