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
    return 0;
}

DPI int sv2c_cosimserver_ep_test(unsigned int endpoint_id, unsigned int* msg_size)
{
    return -1;
}

DPI int sv2c_cosimserver_ep_tryget(unsigned int endpoint_id, const svOpenArrayHandle data, unsigned int* size_bytes)
{

    return -1;
}

DPI int sv2c_cosimserver_ep_tryput(unsigned int endpoint_id, const svOpenArrayHandle data, int data_limit)
{
    return -1;
}

DPI void sv2c_cosimserver_fini()
{
    cout << "dpi_finish" << endl;
    // Stop = true;
    // MainThread.join();
    // if (server != nullptr)
    // {
    //     server->Stop();
    //     server = nullptr;
    // }
}

DPI int sv2c_cosimserver_init()
{
    std::cout << "init()" << std::endl;
    if (server == nullptr)
    {
        server = new RpcServer();
        server->Run(1111);
    }
    return 0;
}