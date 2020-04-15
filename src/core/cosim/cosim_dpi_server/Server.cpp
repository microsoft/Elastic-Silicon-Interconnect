
#include "Server.hpp"
#include <kj/debug.h>
#include <stdexcept>

using namespace std;
using namespace capnp;

kj::Promise<void> CosimServer::list(ListContext context)
{
    auto ifaces = context.getResults().initIfaces((unsigned int)_Reg->EndPoints.size());
    unsigned int ctr = 0u;
    for (auto i = _Reg->EndPoints.begin(); i != _Reg->EndPoints.end(); i++)
    {
        ifaces[ctr].setEndpointID(i->first);
        ifaces[ctr].setTypeID(i->second->GetEsiTypeId());
        ctr++;
    }
    return kj::READY_NOW;
}


kj::Promise<void> CosimServer::open (OpenContext ctxt)
{
    auto epIter = _Reg->EndPoints.find(ctxt.getParams().getIface().getEndpointID());
    KJ_REQUIRE(epIter != _Reg->EndPoints.end(), "Could not find endpoint");

    auto& ep = epIter->second;
    auto gotLock = ep->SetInUse();
    KJ_REQUIRE(gotLock, "Endpoint in use");

    ctxt.getResults().setIface(EsiDpiEndpoint::Client(kj::heap<EndPointServer>(ep)));
    return kj::READY_NOW;
}

RpcServer::~RpcServer()
{
    Stop();
}

void RpcServer::MainLoop(uint16_t port)
{
    _RpcServer = new EzRpcServer(kj::heap<CosimServer>(&EndPoints), "*", port);
    auto& waitScope = _RpcServer->getWaitScope();

    // OK, this is hacky as shit, but it unblocks me and isn't too inefficient
    while (!_Stop)
    {
        waitScope.poll();
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}

void RpcServer::Run(uint16_t port)
{
    if (_MainThread == nullptr) {
        _MainThread = new thread(&RpcServer::MainLoop, this, port);
    } else {
        throw runtime_error("Cannot Run() RPC server more than once!");
    }
}

void RpcServer::Stop()
{
    if (_MainThread == nullptr) {
        throw runtime_error("RpcServer not Run()");
    } else if (!_Stop) {
        _Stop = true;
        _MainThread->join();
    }
}
