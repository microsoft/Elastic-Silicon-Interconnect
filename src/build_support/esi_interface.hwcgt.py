cgt.Name = "esi_interface"
cgt.Description = "Generate SystemVerilog interfaces from a spec using Elastic Silicon Interfaces"

def Generate(run, proj, aq):
    import os
    import os.path as path
    import subprocess

    thisDir = path.dirname(__file__)
    svCodeGenProj = path.normpath(path.join(thisDir, "..", "core", "SVCodeGen"))

    ret = True
    for pf in proj.CollectInputs(aq):
        if pf.File.Name.EndsWith(".capnp"):
            print "%s %s" % (svCodeGenProj, pf.File.FullName)
            outDir = "%s/%s.o" % (os.getcwd(), pf.File.Name)
            if not path.exists(outDir):
                os.mkdir(outDir)
            rc = subprocess.call("dotnet run --project %s -i %s -o %s" % (svCodeGenProj, pf.File.FullName, outDir))
            if rc != 0:
                ret = False
                proj.Log(LogMessage (
                    Level = LogLevel.Error,
                    Message = "SVCodeGen on file '%s' failed with exit code '%s'" % (pf.File.Name, rc)
                ))
            else:
                proj.Glob("%s/*.sv" % outDir, Aspect.Common)
                proj.Glob("%s/*.svh" % outDir, Aspect.Common)
    return ret

cgt.Generate = Generate
