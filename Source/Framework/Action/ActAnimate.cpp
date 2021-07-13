#include "ActAnimate.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/Module.h"
#include "Framework/Framework/NumericalModel.h"
#include "Framework/Framework/ControllerAnimation.h"
#include "Framework/Framework/CollisionModel.h"
#include "Framework/Framework/TopologyMapping.h"
#include "Framework/Framework/ModuleCustom.h"

namespace PhysIKA {

AnimateAct::AnimateAct(float dt)
{
    m_dt = dt;
}

AnimateAct::~AnimateAct()
{
}

void AnimateAct::process(Node* node)
{
    if (node == NULL)
    {
        Log::sendMessage(Log::Error, "Node is invalid!");
        return;
    }
    if (node->isActive())
    {
        node->updateStatus();

        auto customModules = node->getCustomModuleList();
        for (std::list<std::shared_ptr<CustomModule>>::iterator iter = customModules.begin(); iter != customModules.end(); iter++)
        {
            (*iter)->update();
        }

        node->advance(node->getDt());
        node->updateTopology();

        /*if (node->getAnimationController() != nullptr)
            {
                node->getAnimationController()->execute();
            }
            else
            {
                auto nModel = node->getNumericalModel();
                if (nModel == NULL)
                {
                    Log::sendMessage(Log::Warning, node->getName() + ": No numerical model is set!");
                }
                else
                {
                    nModel->step(node->getDt());
                    nModel->updateTopology();
                }
                auto cModels = node->getCollisionModelList();
                for (std::list<std::shared_ptr<CollisionModel>>::iterator iter = cModels.begin(); iter != cModels.end(); iter++)
                {
                    (*iter)->doCollision();
                }
            }*/
    }

    //         if (node->getAnimationController())
    //         {
    //             node->getAnimationController()->execute();
    //         }
}

}  // namespace PhysIKA