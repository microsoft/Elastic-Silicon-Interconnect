
# The Elastic Silicon Interconnect Project

Long ago, software function calling conventions were ad-hoc. This led to
issues, particularly with register clobbering and stack corruption. This is --
in large part -- the state of FPGA/ASIC design today: wire signaling protocols
are often ad-hoc, which also leads to major issues. Though there are efforts
to standardize the signaling protocols there are many minor and major
variants, both of which lead to confusion which can cause real problems when
one is listening to and twiddling the wires manually. ESI solves this by
providing a standardized interface to developers then figures out the
signaling details and conversions between then.

While the ABI/signaling problem is slowly being partially solved, it does not
speak to the types of data on the wires – the software analogy being memory
and registers. In the software world, data types were added. More and more
complex type systems began to evolve – to great successes in some cases as
strong typing can help developers avoid bugs and assist in debugging. In the
FPGA/ASIC world, RTL-level languages are starting to get basic types but
across interconnects it is still common for the data types to be informally
specified in a data sheet. This indicates a total failure of the basic type
system which RTL supports.

The Elastic Silicon Interconnect (ESI) project raises the bar on both fronts. On the
data type front, it defines a rich, hardware-centric type system to allow
more formal data type definitions and strong static type safety. On the
ABI/signaling front, it can build simple, latency-insensitive interfaces and
abstract away the signaling protocol. Essentially, the intent is to cleanly
separate/abstract the physical signaling layer from the message layer. This
enables many tasks to be automated including – but not limited to – the
following:

1) Inter-language communication
2) Type checking to reduce bugs at interface boundaries
3) Correct-by-construction building of communication fabric (including clock
domain crossings)
4) Automated decision making about the physical signaling between modules
5) Automated software API generation which bridges over PCIe, network, or
simulation
6) Automatic pipelining based on floor planning between modules to reduce
timing closure pressure
7) Compatibility between modules with different bandwidths (automatic
gearboxing)
8) Type and signal aware debuggers/monitors in communication fabric
9) Common interface for board support packages
10) Extensible services to support global resources (e.g. telemetry)

# Project Status

The ESI project is in its infancy -- it is not complete by any means. We are
always looking for people to experiment with it and contribute!

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
