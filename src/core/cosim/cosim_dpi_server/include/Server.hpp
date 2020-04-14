#include "esi_cosim_dpi.capnp.h"
#include "EndPoint.hpp"
#include <kj/async.h>
#include <map>

#ifndef __COSIM_DPI_SERVER_HPP__
#define __COSIM_DPI_SERVER_HPP__

class EndPointServer : public EsiDpiInterface::Server
{
    EndPoint* _EndPoint;
public:

    EndPointServer(EndPoint* ep) :
        _EndPoint(ep)
    { }

    EndPoint* GetEndPoint()
    {
        return _EndPoint;
    }
};

class CosimServer : public CosimDpiServer::Server
{
    std::map<int, EndPoint*> _Endpoints;

public:
    CosimServer()
    { }

    kj::Promise<void> list(ListContext ctxt);

    kj::Promise<void> open (OpenContext ctxt);

    void RegisterEndPoint(int ep_id, EndPoint* ep)
    {
        _Endpoints[ep_id] = ep;
        // kj::Own<EndPointServer>& epsCurr = _Endpoints[ep_id];
        // epsCurr = kj::Own<EndPointServer>(new EndPointServer(ep), kj::DestructorOnlyDisposer<EndPointServer>::instance);
    }
};

#endif