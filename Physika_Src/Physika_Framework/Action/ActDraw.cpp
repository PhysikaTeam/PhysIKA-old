#include "ActDraw.h"
#include "Framework/ModuleVisual.h"

namespace Physika
{
	
	DrawAct::DrawAct()
	{

	}

	DrawAct::~DrawAct()
	{

	}

	void DrawAct::Process(Node* node)
	{
		std::list<VisualModule*>& list = node->getVisualModuleList();
		std::list<VisualModule*>::iterator iter;
		for (iter = list.begin(); iter != list.end(); iter++)
		{
			(*iter)->display();
		}
	}
}