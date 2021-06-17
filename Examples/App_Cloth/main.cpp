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

#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Rendering/PointRenderModule.h"
#include "ParticleCloth.h"


#include "Framework/Topology/TriangleSet.h"
#include "Rendering/RigidMeshRender.h"


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
	root->loadCube(Vector3f(-0.1f, 0.0f, -1.0f), Vector3f(1.1f, 2.0f, 1.1f), 0.02f, true);
	root->loadShpere(Vector3f(0.5), 0.2f, 0.01f, false, true);
	{
		std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
		root->addRigidBody(rigidbody);
		rigidbody->loadShape("../../Media/standard/standard_sphere.obj");
		rigidbody->setActive(false);
		auto rigidTri = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(rigidbody->getSurface()->getTopologyModule());
		rigidTri->scale(0.08f);
		rigidTri->translate(Vector3f(0.5));

		auto renderModule = std::make_shared<RigidMeshRender>(rigidbody->getTransformationFrame());
		renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
		rigidbody->getSurface()->addVisualModule(renderModule);
	}

	for (int i = 0; i < 5; i++)
	{
		std::shared_ptr<ParticleCloth<DataType3f>> child3 = std::make_shared<ParticleCloth<DataType3f>>();
		root->addParticleSystem(child3);

		auto m_pointsRender = std::make_shared<PointRenderModule>();

		m_pointsRender->setColor(Vector3f(1-0.2*i, 0.2*i, 1));
		child3->addVisualModule(m_pointsRender);
		child3->setVisible(true);

		child3->setMass(1.0);
		child3->loadParticles("../../Media/cloth/clothLarge.obj");
		child3->loadSurface("../../Media/cloth/clothLarge.obj");

		child3->translate(Vector3f(0.0f, 0.8f + 0.02*i, 0.0f));
	}

	std::shared_ptr<ParticleCloth<DataType3f>> child3 = std::make_shared<ParticleCloth<DataType3f>>();
	root->addParticleSystem(child3);

	auto m_pointsRender = std::make_shared<PointRenderModule>();

	m_pointsRender->setColor(Vector3f(1, 0.2, 1));
	child3->addVisualModule(m_pointsRender);
	child3->setVisible(true);

	child3->setMass(1.0);
  	child3->loadParticles("../../Media/cloth/clothLarge.obj");
  	child3->loadSurface("../../Media/cloth/clothLarge.obj");

	child3->translate(Vector3f(0.0f, 0.8f, 0.0f));
}


int main()
{
	int* ptr;
	cuSafeCall(cudaMalloc((void**)&ptr, 4 * 1000));

	DeviceArray<Vector3f> cd;
	cd.resize(1000);

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


