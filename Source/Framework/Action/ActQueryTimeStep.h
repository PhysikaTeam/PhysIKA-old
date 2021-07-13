#pragma once
#include "Action.h"

namespace PhysIKA {
class QueryTimeStep : public Action
{
public:
    QueryTimeStep();
    virtual ~QueryTimeStep();

    float getTimeStep();
    void  reset();

private:
    void process(Node* node) override;

    float m_timestep;
};
}  // namespace PhysIKA