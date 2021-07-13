#ifndef FRAMEWORK_RESET_H
#define FRAMEWORK_RESET_H

#include "Action.h"

namespace PhysIKA {
class ResetAct : public Action
{
public:
    ResetAct();
    virtual ~ResetAct();

private:
    void process(Node* node) override;
};
}  // namespace PhysIKA

#endif
