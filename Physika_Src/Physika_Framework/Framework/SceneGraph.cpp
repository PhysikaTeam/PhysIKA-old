#include "SceneGraph.h"
#include "Action/ActAnimate.h"
#include "Action/ActDraw.h"
#include "Action/ActInit.h"
#include "Framework/SceneLoaderFactory.h"

namespace Physika
{
SceneGraph& SceneGraph::getInstance()
{
	static SceneGraph m_instance;
	return m_instance;
}

bool SceneGraph::initialize()
{
	if (m_initialized)
	{
		return true;
	}
	//TODO: check initialization
	m_root->traverse<InitAct>();
	m_initialized = true;

	return m_initialized;
}

void SceneGraph::draw()
{
	m_root->traverse<DrawAct>();
}

void SceneGraph::advance(float dt)
{
//	AnimationController*  aController = m_root->getAnimationController();
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
	SceneLoader* loader = SceneLoaderFactory::getInstance().getEntryByFileName(name);
	if (loader)
	{
		m_root = loader->load(name);
		return true;
	}

	return false;
}

}