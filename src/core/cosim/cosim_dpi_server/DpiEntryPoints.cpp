#include "Server.hpp"
#include "dpi.hpp"

#include <algorithm>

using namespace std;

RpcServer* server = nullptr;

//check that an array is an array of bytes and has some size
static int validate_sv_open_array(const svOpenArrayHandle data, int expected_elem_size) {
    int err = 0;
    if (svDimensions(data) != 1) {
        printf("DPI-C: ERROR passed array argument that doesn't have expected 1D dimensions\n");
        err++;
    }
    if (svGetArrayPtr(data) == NULL) {
        printf("DPI-C: ERROR passed array argument that doesn't have C layout (ptr==NULL)\n");
        err++;
    }
    int total_bytes = svSizeOfArray(data);
    if (total_bytes == 0) {
        printf("DPI-C: ERROR passed array argument that doesn't have C layout (total_bytes==0)\n");
        err++;
    }
    int num_elems = svSize(data, 1);
    int elem_size = num_elems == 0 ? 0 : (total_bytes / num_elems);
    if (num_elems * expected_elem_size != total_bytes) {
        printf("DPI-C: ERROR: passed array argument that doesn't have expected element-size: expected=%d actual=%d\n",
                expected_elem_size, elem_size);
    }
    return err;
}

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

DPI int sv2c_cosimserver_ep_tryget(unsigned int endpoint_id, const svOpenArrayHandle data, unsigned int* data_limit)
{
    if (server == nullptr)
        return -3;

    if (validate_sv_open_array(data, sizeof(int8_t)) != 0) {
        printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array", __func__, __LINE__);
        return -1;
    }

    if (*data_limit == ~0) { //used default param
        *data_limit = svSizeOfArray(data);
    }
  
    if (*data_limit > (unsigned)svSizeOfArray(data)) {
        printf("ERROR: DPI-func=%s line %d event=invalid-size", __func__,__LINE__);
        return -2;
    }

    try
    {
        EndPoint::BlobPtr msg;
        if (server->EndPoints[endpoint_id]->GetMessageToSim(msg)) {
            uint8_t* databuf = (uint8_t*)svGetArrayPtr(data);
            if (msg->size() > *data_limit)
            {
                printf("ERROR: Message size too big to fit in RTL buffer");
                return -4;
            }

            copy_n(msg->data(), msg->size(), databuf);
            memset(databuf + msg->size(), 0, *data_limit - msg->size());
            // *data_limit = msg->size();
            return 0;
        } else {
            *data_limit = 0;
            return 0;
        }
    }
    catch (const runtime_error& rt)
    {
        cout << rt.what() << endl;
        return -5;
    }
}

DPI int sv2c_cosimserver_ep_tryput(unsigned int endpoint_id, const svOpenArrayHandle data, int data_limit)
{
    if (server == nullptr)
        return -1;

    if (validate_sv_open_array(data, sizeof(int8_t)) != 0) {
        printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array", __func__, __LINE__);
        return -3;
    }

    if (data_limit < 0) { // used default param
        data_limit = svSizeOfArray(data);
    }
    if (data_limit > svSizeOfArray(data)) { // not enough data
        printf("ERROR: DPI-func=%s line %d event=invalid-size limit %d array %d\n", __func__,__LINE__, data_limit, svSizeOfArray(data));
        return -2;
    }

    try
    {
        uint8_t* databuf = (uint8_t*)svGetArrayPtr(data);
        EndPoint::BlobPtr blob = make_shared<EndPoint::Blob>(databuf, databuf + data_limit);
        server->EndPoints[endpoint_id]->PushMessageToClient(blob);
        return 0;
    }
    catch(const runtime_error& e)
    {
        cout << e.what() << '\n';
        return -4;
    }
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