#include "ActDraw.h"
#include "Physika_Framework/Framework/ModuleVisual.h"

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
}