
#include "Server.hpp"
#include <kj/debug.h>
#include <stdexcept>

using namespace std;
// CosimServer* server = nullptr;

capnp::EzRpcServer* RpcServer;
thread MainThread;

void MainLoop(uint16_t port)
{
    RpcServer = new capnp::EzRpcServer(kj::heap<CosimServer>(), "*", port);
    auto& waitScope = RpcServer->getWaitScope();
    kj::NEVER_DONE.wait(waitScope);
}

void Run(uint16_t port)
{
    // if (RpcServer != nullptr)
    // {
    //     throw std::runtime_error("Cannot start cosim rpc server twice!");
    // }
    MainThread = thread(MainLoop, port);
}

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

    ctxt.getResults().setIface(EsiDpiEndpoint::Client(kj::heap<EndPointServer>(ep)));
    return kj::READY_NOW;
}