#pragma once

#include "Framework/Framework/Module.h"

namespace PhysIKA {

class RigidDebugInfoModule : public Module
{
    DECLARE_CLASS(RigidDebugInfoModule)

public:
public:
    RigidDebugInfoModule() {}

    bool initialize() {}

    virtual void begin() {}

    virtual bool execute();

    virtual void end() {}
};

}  // namespace PhysIKA