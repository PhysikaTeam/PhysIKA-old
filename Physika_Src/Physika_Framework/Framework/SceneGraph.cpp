#include "SceneGraph.h"
#include "Action/ActAnimate.h"
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
	return false;
}

void SceneGraph::draw()
{

}

void SceneGraph::advance(float dt)
{
	std::shared_ptr<AnimationController>  aController = m_root->getAnimationController();
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