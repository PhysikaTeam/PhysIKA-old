#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Physika_GUI/GlutGUI/GLApp.h"

#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Log.h"

#include "Physika_Render/PointRenderModule.h"

#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include "Physika_Dynamics/ParticleSystem/Peridynamics.h"

#include "Physika_Framework/Collision/CollidableSDF.h"
#include "Physika_Framework/Collision/CollidablePoints.h"
#include "Physika_Framework/Collision/CollisionSDF.h"
#include "Physika_Dynamics/RigidBody/RigidBody.h"
#include "Physika_Framework/Framework/Gravity.h"
#include "Physika_Dynamics/ParticleSystem/FixedPoints.h"
#include "Physika_Framework/Collision/CollisionPoints.h"

using namespace std;
using namespace Physika;

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<Node> root = scene.createNewScene<Node>("root");
	auto collidable1 = std::make_shared<CollidableSDF<DataType3f>>();
	root->setCollidableObject(collidable1);

	// 	std::shared_ptr<Node> c1 = root->createChild<Node>("child1");
	// 	auto* pSet1 = new PointSet<Vector3f>();
	// 	c1->setTopologyModule(pSet1);

	//create root node
	std::shared_ptr<Node> c1 = root->createChild<Node>("child1");

	auto pSet = std::make_shared<PointSet<DataType3f>>();
	c1->setTopologyModule(pSet);

	auto pS1 = std::make_shared<Peridynamics<DataType3f>>();
	//	auto pS1 = std::make_shared<RigidBody<DataType3f>>();
	c1->setNumericalModel(pS1);

	auto render = std::make_shared<PointRenderModule>();
	//	render->setVisible(false);
	c1->addVisualModule(render);

	auto cPoints = std::make_shared<CollidablePoints<DataType3f>>();
	c1->setCollidableObject(cPoints);

	auto gravity = std::make_shared<Gravity<DataType3f>>();
	c1->addForceModule(gravity);

	//	auto fixed = std::make_shared<FixedPoints<DataType3f>>();
	//	fixed->addPoint(0);
	//	c1->addConstraintModule(fixed);

	auto cModel = std::make_shared<CollisionSDF<DataType3f>>();
	cModel->setCollidableSDF(collidable1);
	cModel->addCollidableObject(cPoints);
//	cModel->setSDF(collidable1->getSDF());
	root->addCollisionModel(cModel);

	auto pModel = std::make_shared<CollisionPoints<DataType3f>>();
	pModel->addCollidableObject(cPoints);
	root->addCollisionModel(pModel);
}


int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(800, 600);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}

