#include "Framework/ModuleRender.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(RenderModule)

RenderModule::RenderModule()
{
}

RenderModule::~RenderModule()
{
}

bool RenderModule::insertToNode(Node* node)
{
	node->addRenderModule(this);
	return true;
}

bool RenderModule::deleteFromNode(Node* node)
{
	node->deleteRenderModule(this);
	return true;
}

}