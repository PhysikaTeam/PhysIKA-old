#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/RigidBody.h"

#include "Rendering/PointRenderModule.h"

#include "ParticleViscoplasticBody.h"

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

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

	for (size_t i = 0; i < 5; i++)
	{
		root->loadCube(Vector3f(0.2 + i * 0.08, 0.2, 0), Vector3f(0.25 + i * 0.08, 0.25, 1), 0.005, false, true);
	}

	std::shared_ptr<ParticleViscoplasticBody<DataType3f>> child3 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(0, 1, 1));
	child3->addVisualModule(ptRender);

	child3->setMass(1.0);
  	child3->loadParticles("../../Media/bunny/bunny_points.obj");
  	child3->loadSurface("../../Media/bunny/bunny_mesh.obj");
	child3->translate(Vector3f(0.4, 0.4, 0.5));

	std::shared_ptr<ParticleViscoplasticBody<DataType3f>> child4 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
	root->addParticleSystem(child4);
	auto ptRender2 = std::make_shared<PointRenderModule>();
	ptRender2->setColor(Vector3f(1, 0, 1));
	child4->addVisualModule(ptRender2);

	child4->setMass(1.0);
	child4->loadParticles("../../Media/bunny/bunny_points.obj");
	child4->loadSurface("../../Media/bunny/bunny_mesh.obj");
	child4->translate(Vector3f(0.4, 0.4, 0.9));
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


