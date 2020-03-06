using Serilog;
using Serilog.Core;

using System;

namespace Esi
{
    public class EsiContext : IDisposable
    {
        public Logger? Log { get; protected set; }

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

        public void Dispose()
        {
            Log?.Dispose();
            Log = null;
        }
    }
}