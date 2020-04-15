#include "esi_cosim_dpi.capnp.h"
#include <queue>
#include <map>
#include <memory>
#include <stdexcept>

#ifndef __ESI_ENDPOINT_HPP__
#define __ESI_ENDPOINT_HPP__

class EndPoint
{
    uint64_t _EsiTypeId;
    int      _MaxSize;
    bool     _InUse;

    std::queue<std::shared_ptr<capnp::Data> > toCosim;
    std::queue<std::shared_ptr<capnp::Data> > toClient;

public:
    EndPoint(uint64_t EsiTypeId, int MaxSize) :
        _EsiTypeId(EsiTypeId),
        _MaxSize(MaxSize),
        _InUse(false)
    {    }

    uint64_t GetEsiTypeId()
    {
        return _EsiTypeId;
    }

    bool SetInUse()
    {
        if (_InUse)
            return false;
        _InUse = true;
        return true;
    }

    void PushMessageToSim(std::shared_ptr<capnp::Data> msg)
    {
        toCosim.push(msg);
    }

    bool GetMessageToSim(std::shared_ptr<capnp::Data>& msg)
    {
        if (toCosim.size() > 0)
        {
            msg = toCosim.front();
            toCosim.pop();
            return true;
        }
        return false;
    }

    void PushMessageToClient(std::shared_ptr<capnp::Data> msg)
    {
        toClient.push(msg);
    }

    bool GetMessageToClient(std::shared_ptr<capnp::Data>& msg)
    {
        if (toClient.size() > 0)
        {
            msg = toClient.front();
            toClient.pop();
            return true;
        }
        return false;
    }
};

class EndPointRegistry
{
public:
    std::map<int, std::unique_ptr<EndPoint> > EndPoints;

    ~EndPointRegistry();

    /// Takes ownership of ep
    void RegisterEndPoint(int ep_id, long long esi_type_id, int type_size);

    std::unique_ptr<EndPoint>& operator[](int ep_id)
    {
        auto& ep = EndPoints.find(ep_id);
        if (ep == EndPoints.end())
            throw std::runtime_error("Could not find endpoint");
        return ep->second;
    }
};

#endif
