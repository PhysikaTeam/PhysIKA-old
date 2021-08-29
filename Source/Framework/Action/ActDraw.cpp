#include "ActDraw.h"
#include "Framework/Framework/ModuleVisual.h"

namespace PhysIKA {

DrawAct::DrawAct()
{
}

DrawAct::~DrawAct()
{
}

void DrawAct::process(Node* node)
{
    if (!node->isVisible())
    {
        return;
    }

    auto& list = node->getVisualModuleList();
    for (std::list<std::shared_ptr<VisualModule>>::iterator iter = list.begin(); iter != list.end(); iter++)
    {
        if ((*iter)->isVisible())
        {
            (*iter)->updateRenderingContext();
            (*iter)->display();
        }
    }
}
}  // namespace PhysIKA