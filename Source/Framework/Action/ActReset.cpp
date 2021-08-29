#include "ActReset.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {

ResetAct::ResetAct()
{
}

ResetAct::~ResetAct()
{
}

void ResetAct::process(Node* node)
{
    if (node == NULL)
    {
        Log::sendMessage(Log::Error, "Node is invalid!");
        return;
    }

    node->resetStatus();
}

}  // namespace PhysIKA