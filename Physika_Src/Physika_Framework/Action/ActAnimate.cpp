#include "ActAnimate.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Framework/Framework/NumericalModel.h"

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
		//node->advance(node->getDt());
		std::list<Module*> list = node->getModuleList();
		std::list<Module*>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->execute();
		}
//		node->getNumericalModel()->execute2();
	}

}