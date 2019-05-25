#include "ActInit.h"
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Framework/Framework/NumericalModel.h"

namespace Physika
{
	InitAct::InitAct()
	{

	}

	InitAct::~InitAct()
	{

	}

	void InitAct::Process(Node* node)
	{
		node->resetStatus();
		node->initialize();

		auto& list = node->getModuleList();
		std::list<std::shared_ptr<Module>>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->initialize();
		}
	}

}