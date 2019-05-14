#include "ActAnimate.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Framework/Framework/NumericalModel.h"
#include "Physika_Framework/Framework/ControllerAnimation.h"
#include "Physika_Framework/Framework/CollisionModel.h"
#include "Physika_Framework/Framework/TopologyMapping.h"

namespace Physika
{
	
	AnimateAct::AnimateAct()
	{

	}

	AnimateAct::~AnimateAct()
	{

	}

	void AnimateAct::Process(Node* node)
	{
		if (node == NULL)
		{
			Log::sendMessage(Log::Error, "Node is invalid!");
			return;
		}
		if (node->isActive())
		{
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


// 		if (node->getAnimationController())
// 		{
// 			node->getAnimationController()->execute();
// 		}
	}

}