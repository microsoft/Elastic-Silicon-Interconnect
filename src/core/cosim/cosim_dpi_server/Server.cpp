
#include "Server.hpp"
#include <kj/debug.h>

kj::Promise<void> CosimServer::list(ListContext context)
{
    auto ifaces = context.getResults().initIfaces((unsigned int)_Endpoints.size());
    unsigned int ctr = 0u;
    for (auto i = _Endpoints.begin(); i != _Endpoints.end(); i++)
    {
        ifaces[ctr].setTypeID(i->second->GetEsiTypeId());
        ifaces[ctr].setEndpointID(i->first);
    }
    context.getResults().setIfaces(ifaces);
    return kj::READY_NOW;
}


kj::Promise<void> CosimServer::open (OpenContext ctxt)
{
    auto epIter = _Endpoints.find(ctxt.getParams().getIface().getEndpointID());
    KJ_REQUIRE(epIter != _Endpoints.end(), "Could not find endpoint");

    auto ep = epIter->second;
    auto inUse = !ep->SetInUse();
    KJ_REQUIRE(inUse, "Endpoint in use");

    EsiDpiInterface::Client client(kj::Own<EsiDpiInterface::Server>(
        new EndPointServer(ep),
        kj::DestructorOnlyDisposer<EsiDpiInterface>::instance
    ));
    ctxt.getResults().setIface(client);
    return kj::READY_NOW;
}