#include "esi_cosim_dpi.capnp.h"

class CosimServer : public CosimDpiServer::Server
{
    public ::kj::Promise<void> list(ListContext context)
    {

    }
}