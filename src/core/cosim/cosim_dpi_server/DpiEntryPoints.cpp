#include "Server.hpp"
#include "dpi.hpp"

using namespace std;

RpcServer* server = nullptr;

// DPI entry points

DPI int sv2c_cosimserver_conn_connected(unsigned int endpoint_id)
{
    return 1;
}

DPI int sv2c_cosimserver_ep_register(int endpoint_id, long long esi_type_id, int type_size)
{
    sv2c_cosimserver_init();
    try
    {
        server->EndPoints.RegisterEndPoint(endpoint_id, esi_type_id, type_size);
        return 0;
    }
    catch (const runtime_error& rt)
    {
        cerr << rt.what() << endl;
        return 1;
    }
}

DPI int sv2c_cosimserver_ep_test(unsigned int endpoint_id, unsigned int* msg_size)
{
    return -1;
}

DPI int sv2c_cosimserver_ep_tryget(unsigned int endpoint_id, const svOpenArrayHandle data, unsigned int* size_bytes)
{
    if (server == nullptr)
        return -1;

    try
    {
        EndPoint::BlobPtr msg;
        if (server->EndPoints[endpoint_id]->GetMessageToSim(msg)) {
            return -1;
        } else {
            size_bytes = 0;
            return 0;
        }
    }
    catch (const runtime_error& rt)
    {
        cerr << rt.what() << endl;
        return 1;
    }
}

DPI int sv2c_cosimserver_ep_tryput(unsigned int endpoint_id, const svOpenArrayHandle data, int data_limit)
{
    if (server == nullptr)
        return -1;

    return -1;
}

DPI void sv2c_cosimserver_fini()
{
    cout << "dpi_finish" << endl;
    if (server != nullptr)
    {
        server->Stop();
        server = nullptr;
    }
}

DPI int sv2c_cosimserver_init()
{
    if (server == nullptr)
    {
        std::cout << "init()" << std::endl;
        server = new RpcServer();
        server->Run(1111);
    }
    return 0;
}