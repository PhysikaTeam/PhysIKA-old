#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/ParticleSystem/HeightField.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"

#include "Rendering/PointRenderModule.h"

using namespace std;
using namespace PhysIKA;


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
	scene.setUpperBound(Vector3f(1.5, 1, 1.5));
	scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	//root->loadCube(Vector3f(-0.5, 0, -0.5), Vector3f(1.5, 2, 1.5), 0.02, true);
	//root->loadSDF("../../Media/bowl/bowl.sdf", false);

	std::shared_ptr<HeightField<DataType3f>> child1 = std::make_shared<HeightField<DataType3f>>();
	root->addParticleSystem(child1);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(1, 0, 0));
	ptRender->setColorRange(0, 4);
	child1->addVisualModule(ptRender);

	//child1->loadParticles("../Media/fluid/fluid_point.obj");
	child1->loadParticles(Vector3f(0, 0.2, 0), Vector3f(1, 1.5, 1), 0.01, 0.1);
	child1->setMass(100);
	child1->getVelocity()->connect(ptRender->m_vecIndex);

	//std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	//root->addRigidBody(rigidbody);
	//rigidbody->loadShape("../../Media/bowl/bowl.obj");
	//rigidbody->setActive(false);
}

int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}