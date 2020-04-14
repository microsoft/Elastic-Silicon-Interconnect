#include "esi_cosim_dpi.capnp.h"
#include <queue>

#ifndef __ESI_ENDPOINT_HPP__
#define __ESI_ENDPOINT_HPP__

class EndPoint
{
    uint64_t EsiTypeId;
    bool _InUse;

    std::queue<capnp::Data*> toCosim;
    std::queue<capnp::Data*> toClient;

public:
    EndPoint(uint64_t EsiTypeId) :
        EsiTypeId(EsiTypeId),
        _InUse(false)
    {    }

    uint64_t GetEsiTypeId()
    {
        return EsiTypeId;
    }

    bool SetInUse()
    {
        if (_InUse)
            return false;
        _InUse = true;
        return true;
    }

    void PushMessageToSim(capnp::Data*& msg)
    {
        toCosim.push(msg);
    }

    bool GetMessageToSim(capnp::Data*& msg)
    {
        if (toCosim.size() > 0)
        {
            msg = toCosim.front();
            toCosim.pop();
            return true;
        }
        return false;
    }

    void PushMessageToClient(capnp::Data* msg)
    {
        toClient.push(msg);
    }

    bool GetMessageToClient(capnp::Data*& msg)
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

#endif
