#include "esi_cosim_dpi.capnp.h"
#include "EndPoint.hpp"
#include <capnp/ez-rpc.h>
#include <kj/async.h>
#include <map>
#include <thread>
#include <iostream>

#ifndef __COSIM_DPI_SERVER_HPP__
#define __COSIM_DPI_SERVER_HPP__

class EndPointServer : public EsiDpiEndpoint::Server
{
    std::unique_ptr<EndPoint>& _EndPoint;
public:

    EndPointServer(std::unique_ptr<EndPoint>& ep) :
        _EndPoint(ep)
    { }

    std::unique_ptr<EndPoint>& GetEndPoint()
    {
        return _EndPoint;
    }
};

class CosimServer : public CosimDpiServer::Server
{
    EndPointRegistry* _Reg;

public:
    CosimServer(EndPointRegistry* reg) :
        _Reg(reg)
    { }

    kj::Promise<void> list(ListContext ctxt);
    kj::Promise<void> open (OpenContext ctxt);
};

class RpcServer
{
    capnp::EzRpcServer* _RpcServer;
    std::thread* _MainThread;
    volatile bool _Stop;

    void MainLoop(uint16_t port);

public:
    EndPointRegistry EndPoints;

    RpcServer() :
        _RpcServer(nullptr),
        _MainThread(nullptr),
        _Stop(false)
    {   }

    ~RpcServer();

    void Run(uint16_t port);
    void Stop();
};

#endif