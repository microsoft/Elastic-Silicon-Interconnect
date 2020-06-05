// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Serilog;
using Serilog.Core;
using Serilog.Events;
using System;

namespace Esi
{
    public class EsiContext : IDisposable, ILogEventSink
    {
        public Logger? Log { get; protected set; }
        public ulong Errors { get; protected set; } = 0;
        public ulong Fatals { get; protected set; } = 0;
        public void ClearCounts() {
            Errors = 0;
            Fatals = 0;
        }

        public bool Failed => Errors > 0 || Fatals > 0;

        public EsiContext(LoggerConfiguration? loggerConfiguration = null)
        {
            if (loggerConfiguration == null)
            {
                loggerConfiguration = new LoggerConfiguration()
                    .MinimumLevel.Debug()
                    .WriteTo.Console();
            }
            Log = loggerConfiguration
                .WriteTo.Sink(this)
                .CreateLogger();
        }

        public void Dispose()
        {
            Log = null;
        }

        public void Emit(LogEvent evt)
        {
            switch (evt.Level)
            {
                case LogEventLevel.Error:
                    Errors ++;
                    break;
                case LogEventLevel.Fatal:
                    Fatals ++;
                    break;
                default:
                    break;
            }
        }
    }
}