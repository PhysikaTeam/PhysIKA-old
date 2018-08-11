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
		std::list<Module*> list = node->getModuleList();
		std::list<Module*>::iterator iter = list.begin();
		for (; iter != list.end(); iter++)
		{
			(*iter)->initialize();
		}
	}

}