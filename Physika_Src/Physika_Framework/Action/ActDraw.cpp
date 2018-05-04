#include "ActDraw.h"
#include "Framework/ModuleRender.h"

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
		std::list<RenderModule*>& list = node->getRenderModuleList();
		std::list<RenderModule*>::iterator iter;
		for (iter = list.begin(); iter != list.end(); iter++)
		{
			(*iter)->Display();
		}
	}
}