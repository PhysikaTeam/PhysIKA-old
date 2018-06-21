#include "SceneGraph.h"
#include "Action/ActAnimate.h"

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

}

void SceneGraph::takeOneFrame()
{
	m_root->traverse<AnimateAct>();
}

void SceneGraph::run()
{

}

}