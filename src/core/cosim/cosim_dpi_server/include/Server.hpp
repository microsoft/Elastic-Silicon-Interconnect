#include "esi_cosim_dpi.capnp.h"
#include "EndPoint.hpp"
#include <capnp/ez-rpc.h>
#include <kj/async.h>
#include <map>
#include <thread>
#include <iostream>

#ifndef __COSIM_DPI_SERVER_HPP__
#define __COSIM_DPI_SERVER_HPP__

void Run(uint16_t port);

class EndPointServer : public EsiDpiEndpoint::Server
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
    }
};

#endif