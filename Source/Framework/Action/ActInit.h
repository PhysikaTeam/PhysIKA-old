#pragma once
#include "Action.h"

namespace PhysIKA {
class InitAct : public Action
{
public:
    InitAct();
    virtual ~InitAct();

private:
    void process(Node* node) override;
};
}  // namespace PhysIKA
