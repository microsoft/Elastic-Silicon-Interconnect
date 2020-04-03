# Frequently Asked Questions

## What are these hwlib.yml and hwcgt.py files?

At Microsoft, we used an internal tool called 'HWBuild' to drive our
synthesis and simulation flows. The 'hwlibs' that you see scattered around
are the declarative definitions which HWBuild consumes. The YAML format,
however, is relatively simple so it would be relatively easy to adapt it to a
simpler tool. The hwcgt.py you see are HWBuild plugins which take care of
code generation -- they are intended for this exactly.

We are considering open sourcing HWBuild at some point, but for the time
being its used for our internal tests which also require tool licenses.
