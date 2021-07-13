#include "ActNodeInfo.h"

namespace PhysIKA {

NodeInfoAct::NodeInfoAct()
{
}

NodeInfoAct::~NodeInfoAct()
{
}

void NodeInfoAct::process(Node* node)
{
    std::cout << node->getName() << std::endl;
}

}  // namespace PhysIKA