#pragma once
#include "Action.h"

namespace PhysIKA {
class PostProcessing : public Action
{
public:
    PostProcessing();
    virtual ~PostProcessing();

private:
    void process(Node* node) override;
};
}  // namespace PhysIKA
