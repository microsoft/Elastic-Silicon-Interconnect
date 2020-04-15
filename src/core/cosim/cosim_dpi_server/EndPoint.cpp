#include "EndPoint.hpp"
#include <iostream>

using namespace std;

EndPointRegistry::~EndPointRegistry()
{
    EndPoints.clear();
}

void EndPointRegistry::RegisterEndPoint(int ep_id, long long esi_type_id, int type_size)
{
    if (EndPoints.find(ep_id) != EndPoints.end())
    {
        throw runtime_error("Endpoint ID already exists!");
    }
    EndPoints[ep_id] = make_unique<EndPoint>(esi_type_id, type_size);
}