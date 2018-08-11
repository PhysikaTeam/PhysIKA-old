#include "SceneGraph.h"
#include "Action/ActAnimate.h"
#include "Action/ActDraw.h"
#include "Action/ActInit.h"
#include "Framework/SceneLoaderFactory.h"

namespace Physika
{
std::shared_ptr<SceneGraph> SceneGraph::m_instance = nullptr;

std::shared_ptr<SceneGraph> SceneGraph::getInstance()
{
	if (m_instance == NULL)
	{
		m_instance = TypeInfo::New<SceneGraph>();
	}
	return m_instance;
}

bool SceneGraph::initialize()
{
	m_root->traverse<InitAct>();
	return false;
}

void SceneGraph::draw()
{
	m_root->traverse<DrawAct>();
}

void SceneGraph::advance(float dt)
{
	AnimationController*  aController = m_root->getAnimationController();
	//	aController->
}

void SceneGraph::takeOneFrame()
{
	m_root->traverse<AnimateAct>();
}

void SceneGraph::run()
{

}

bool SceneGraph::load(std::string name)
{
	SceneLoader* loader = SceneLoaderFactory::getInstance()->getEntryByFileName(name);
	if (loader)
	{
		m_root = std::shared_ptr<Node>(loader->load(name));
		return true;
	}

	return false;
}

}